import itertools
import json
import numpy as np
import networkx as nx
from typing import List, Dict, Any

from src.retrieval.base import BaseRetriever
from src.utils.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    DISCRIMINATOR_PROMPT,
    RELATIONSHIP_EXTRACTION_PROMPT,
    QUERY_DECOMPOSITION_PROMPT
)

def parse_delimited_records(raw: str, n_fields: int):
    raw = raw.split("<|COMPLETE|>")[0]
    records = []
    for chunk in raw.split("##"):
        chunk = chunk.strip()
        if not chunk.startswith("("):
            continue
        chunk = chunk.strip("()\n ")
        parts = [p.strip() for p in chunk.split("<|>")]
        if len(parts) == n_fields:
            records.append(tuple(parts))
    return records

class HSGRAG(BaseRetriever):
    def __init__(self, llm, embedder, config: Dict):
        self.llm = llm
        self.embedder = embedder
        self.config = config
        self.graphs = {}
        self.canons = {}
        self.unit_lookups = {}
        
    def _cosine_sim_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return a_n @ b_n.T

    # --- Extraction Phase ---
    def _extract_entities(self, text_unit: str):
        prompt = ENTITY_EXTRACTION_PROMPT.format(input=text_unit)
        raw = self.llm.generate(prompt)
        records = parse_delimited_records(raw, n_fields=5)

        verdict_prompt = DISCRIMINATOR_PROMPT.format(document=text_unit, extracted=raw)
        verdict = self.llm.generate(verdict_prompt)
        
        if "COMPLETE" not in verdict.upper():
            correction_prompt = prompt + f"\n\nNote: a reviewer found this feedback on a previous attempt: {verdict}\nMake sure to address it and output the FULL corrected entity list."
            raw2 = self.llm.generate(correction_prompt)
            records = parse_delimited_records(raw2, n_fields=5)

        entities = []
        for kind, name, etype, doc_rel, desc in records:
            entities.append({
                "kind": kind, "name": name, "type": etype,
                "document_relation": doc_rel, "description": desc,
            })
        return entities

    def _extract_relationships(self, text_unit: str, entities: list):
        entity_set_str = "\n".join(f"({e['kind']}<|>{e['name']}<|>{e['type']}<|>{e['description']})" for e in entities)
        prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(document=text_unit, entity_set=entity_set_str)
        raw = self.llm.generate(prompt)
        records = parse_delimited_records(raw, n_fields=3)
        return [{"source": s, "relationship": r, "target": t} for s, r, t in records]

    # --- Graph Building ---
    def index(self, chunks: List[Dict], embeddings: Any = None):
        """Indexes chunks by document. Expected chunk format: {'unit_id': 'doc1::u0', 'doc_id': 'doc1', 'text': '...'}"""
        # Group by doc_id
        docs = {}
        for c in chunks:
            docs.setdefault(c["doc_id"], []).append(c)
            
        for doc_id, doc_chunks in docs.items():
            self._index_single_document(doc_id, doc_chunks)

    def _index_single_document(self, doc_id: str, text_units: List[Dict]):
        unit_entities = {}
        unit_relations = {}
        
        # Short-distance extraction
        for u in text_units:
            ents = self._extract_entities(u["text"])
            unit_entities[u["unit_id"]] = ents
            unit_relations[u["unit_id"]] = self._extract_relationships(u["text"], ents)

        pair_relations = {}
        for a, b in zip(text_units, text_units[1:]):
            ents = unit_entities[a["unit_id"]] + unit_entities[b["unit_id"]]
            joint_text = a["text"] + "\n" + b["text"]
            pair_relations[(a["unit_id"], b["unit_id"])] = self._extract_relationships(joint_text, ents)

        # Long-distance extraction
        long_relations = self._long_distance_extraction(text_units, unit_entities)

        # Graph Building
        canon, name_lookup = self._link_entities(unit_entities)
        canon = self._summarize_entities(canon)
        G = self._build_graph(canon, name_lookup, unit_relations, pair_relations, long_relations, doc_id)
        
        self.graphs[doc_id] = G
        self.canons[doc_id] = canon
        self.unit_lookups[doc_id] = {u["unit_id"]: u["text"] for u in text_units}

    def _long_distance_extraction(self, text_units: list, unit_entities: dict):
        all_descs, owner = [], []
        for u in text_units:
            for e in unit_entities.get(u["unit_id"], []):
                all_descs.append(e["description"])
                owner.append(u["unit_id"])
        
        if not all_descs: return {}
        vecs = self.embedder.encode(all_descs)
        
        unit_idx_ranges = {}
        for uid in set(owner):
            unit_idx_ranges[uid] = [i for i, o in enumerate(owner) if o == uid]
            
        long_relations = {}
        units = [u["unit_id"] for u in text_units]
        
        for ua, ub in itertools.combinations(units, 2):
            if abs(units.index(ua) - units.index(ub)) <= 1:
                continue
            idx_a, idx_b = unit_idx_ranges.get(ua, []), unit_idx_ranges.get(ub, [])
            if not idx_a or not idx_b: continue
            
            sim_sub = self._cosine_sim_matrix(vecs[idx_a], vecs[idx_b])
            iou = self._semantic_iou(sim_sub)
            
            if iou > self.config.get('semantic_iou_threshold', 0.3):
                ents = unit_entities[ua] + unit_entities[ub]
                text_a = next(u["text"] for u in text_units if u["unit_id"] == ua)
                text_b = next(u["text"] for u in text_units if u["unit_id"] == ub)
                rels = self._extract_relationships(text_a + "\n" + text_b, ents)
                if rels: long_relations[(ua, ub)] = rels
        return long_relations

    def _semantic_iou(self, sim_sub: np.ndarray) -> float:
        if sim_sub.size == 0: return 0.0
        thresh = self.config.get('semantic_iou_sim_threshold', 0.8)
        a_hits = (sim_sub.max(axis=1) > thresh).sum()
        b_hits = (sim_sub.max(axis=0) > thresh).sum()
        denom = sim_sub.shape[0] + sim_sub.shape[1]
        return (a_hits + b_hits) / denom if denom else 0.0

    def _link_entities(self, unit_entities: dict):
        flat = []
        for uid, ents in unit_entities.items():
            for e in ents:
                flat.append({**e, "source_unit": uid})

        named = [e for e in flat if e["kind"] == "code"]
        semantic = [e for e in flat if e["kind"] == "semantic"]
        merged = {}
        
        for e in named:
            key = ("code", e["name"].strip().lower())
            if key not in merged:
                merged[key] = {"name": e["name"], "kind": "code", "type": e["type"], "descriptions": [], "sources": []}
            merged[key]["descriptions"].append(e["description"])
            merged[key]["sources"].append((e["source_unit"], e["document_relation"]))

        if semantic:
            descs = [e["description"] for e in semantic]
            vecs = self.embedder.encode(descs)
            sim = self._cosine_sim_matrix(vecs, vecs)
            n = len(semantic)
            assigned = [-1] * n
            next_id = 0
            for i in range(n):
                if assigned[i] != -1: continue
                assigned[i] = next_id
                for j in range(i + 1, n):
                    if assigned[j] == -1 and sim[i, j] > self.config.get('entity_link_sim_threshold', 0.9):
                        assigned[j] = next_id
                next_id += 1
            for i, e in enumerate(semantic):
                key = ("semantic", assigned[i])
                if key not in merged:
                    merged[key] = {"name": e["name"], "kind": "semantic", "type": e["type"], "descriptions": [], "sources": []}
                merged[key]["descriptions"].append(e["description"])
                merged[key]["sources"].append((e["source_unit"], e["document_relation"]))

        canon, name_lookup = {}, {}
        for key, rec in merged.items():
            cid = f"{rec['kind']}::{rec['name']}"[:120]
            canon[cid] = rec
            name_lookup[rec["name"].strip().lower()] = cid
        return canon, name_lookup

    def _summarize_entities(self, canon: dict):
        for cid, rec in canon.items():
            if len(rec["descriptions"]) == 1:
                rec["summary"] = rec["descriptions"][0]
            else:
                joined = "\n".join(f"- {d}" for d in rec["descriptions"])
                prompt = f"Summarize the following descriptions of the same entity '{rec['name']}' into one concise, complete-sentence description:\n{joined}"
                rec["summary"] = self.llm.generate(prompt)
                
        names = list(canon.keys())
        summaries = [canon[n]["summary"] for n in names]
        vecs = self.embedder.encode(summaries) if summaries else np.zeros((0, 1))
        for n, v in zip(names, vecs):
            canon[n]["embedding"] = v
        return canon

    def _build_graph(self, canon: dict, name_lookup: dict, unit_relations: dict, pair_relations: dict, long_relations: dict, doc_id: str):
        G = nx.MultiDiGraph()
        for cid, rec in canon.items():
            G.add_node(cid, kind="entity", name=rec["name"], type=rec["type"], description=rec["summary"], doc_id=doc_id)
            for uid, doc_rel in rec["sources"]:
                G.add_node(uid, kind="text_unit", doc_id=doc_id)
                G.add_edge(cid, uid, relation=doc_rel)

        def _rel_type(raw: str) -> str:
            raw_u = raw.upper()
            if "IMPLEMENT" in raw_u: return "IMPLEMENT"
            if "CONTAIN" in raw_u: return "CONTAIN"
            return "DEPEND"

        def _add_rels(rel_list):
            for r in rel_list:
                s = name_lookup.get(r["source"].strip().lower())
                t = name_lookup.get(r["target"].strip().lower())
                if s and t and s in canon and t in canon:
                    G.add_edge(s, t, relation=_rel_type(r["relationship"]))

        for rels in list(unit_relations.values()) + list(pair_relations.values()) + list(long_relations.values()):
            _add_rels(rels)
        return G

    # --- Retrieval Phase ---
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        document_list = list(self.graphs.keys())
        prompt = QUERY_DECOMPOSITION_PROMPT.format(document_list=json.dumps(document_list), user_query=query)
        raw = self.llm.generate(prompt)
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
        except Exception:
            data = {"sub_queries": [{"document": d, "sub_query": query} for d in document_list]}

        all_ranked = []
        for sq in data.get("sub_queries", []):
            doc_id, subquery = sq["document"], sq["sub_query"]
            if doc_id not in self.graphs: continue
            
            visited = self._explore_subgraph(subquery, self.graphs[doc_id], self.canons[doc_id])
            ranked_units = self._aggregate_text_units(visited, self.graphs[doc_id])
            all_ranked.extend([(doc_id, u) for u in ranked_units])
            
        # Format the output matching BaseRetriever
        results = []
        # Return unique chunks based on top_k
        seen = set()
        for doc_id, uid in all_ranked:
            if uid in seen: continue
            seen.add(uid)
            text = self.unit_lookups.get(doc_id, {}).get(uid, "")
            if text:
                results.append({"unit_id": uid, "doc_id": doc_id, "text": text})
            if len(results) >= top_k:
                break
        return results

    def _explore_subgraph(self, subquery: str, G, canon: dict):
        entry_num = self.config.get('entry_num', 7)
        max_depth = self.config.get('max_depth', 3)
        
        entity_ids = [n for n, d in G.nodes(data=True) if d.get("kind") == "entity"]
        if not entity_ids: return set()
        embs = np.stack([canon[i]["embedding"] for i in entity_ids])
        qvec = self.embedder.encode([subquery])
        sims = self._cosine_sim_matrix(qvec, embs)[0]
        top_idx = np.argsort(-sims)[:entry_num]
        entries = [entity_ids[i] for i in top_idx]

        visited = set(entries)
        frontier = set(entries)
        for _ in range(max_depth):
            nxt = set()
            for node in frontier:
                for _, nbr, data in G.out_edges(node, data=True):
                    if data.get("relation") in ("DEPEND", "CONTAIN", "IMPLEMENT") and nbr not in visited:
                        nxt.add(nbr)
            visited |= nxt
            frontier = nxt
            if not frontier: break
        return visited

    def _aggregate_text_units(self, visited_entities: set, G):
        unit_score = {}
        unit_has_define = {}
        for eid in visited_entities:
            for _, uid, data in G.out_edges(eid, data=True):
                if G.nodes[uid].get("kind") != "text_unit": continue
                unit_score[uid] = unit_score.get(uid, 0) + 1
                if data.get("relation") == "DEFINE":
                    unit_has_define[uid] = True
                else:
                    unit_has_define.setdefault(uid, False)
        ranked = sorted(unit_score.keys(), key=lambda u: (unit_has_define.get(u, False), unit_score[u]), reverse=True)
        return ranked
