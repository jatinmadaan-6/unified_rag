# HSG-RAG Prompts (from original paper)

ENTITY_EXTRACTION_PROMPT = """You are an embedded system developer skilled in domain specific customization. You are knowledgeable about general domain knowledge about RTOS, MCU, etc.

-Goal-
Given a technical document, first identify all entities followed by the defined types. For each entity, extract the entity name, type, document relationship and description. Return output in English as a single list of all the entities. Use ## as the list delimiter. When finished, output <|COMPLETE|>

-Schema-
For Code Entity:
- entity_name: The literal name mentioned in the document. Must be exactly the same as the name in the document.
- entity_type: Suggest the categories for the entity, one of ["FUNCTION", "STRUCT", "MACRO", "ENUM", "FILE_NAME", "PIN", "REGISTER_NAME", "DATA_TYPE", "UNKNOWN_NAME"].
- document_relation: The relationship between the entity and the document, one of ["DEFINE", "USE"].
- entity_description: The summarized content of the piece of knowledge, each piece of knowledge should be complete sentences.
Format each entity as
("code"<|><entity_name><|><entity_type><|><document_relation><|><entity_description>)

For Semantic Entity:
- entity_name: A noun phrase or a verb phrase that represents a concept or a task.
- entity_type: Suggest the categories for the entity, one of ["TASK", "SYSTEM_MODULE", "OTHER"].
- document_relation: The relationship between the entity and the document, one of ["DEFINE", "USE"].
- entity_description: The summarized content of the piece of knowledge, each piece of knowledge should be complete sentences.
Format each entity as
("semantic"<|><entity_name><|><entity_type><|><document_relation><|><entity_description>)

#####################
-Examples-
#####################
input document:
esp_err_t esp_spp_init(esp_spp_mode_t mode)
This function is called to ...
#####################
output:
("code"<|>esp_spp_init<|>FUNCTION<|>DEFINE<|>This is a function called to initialize the SPP module.)##
("semantic"<|>Initializing SPP module<|>TASK<|>USE<|>This represents the process of initializing the SPP module.)##
<|COMPLETE|>

#####################
-Real Data-
#####################
input document: {input}
#####################
output:
"""

DISCRIMINATOR_PROMPT = """You previously extracted the following entities from a technical
document based on a schema. Judge whether the extraction is COMPLETE or MISSING important
entities/relations. If missing, briefly say what kind of entity/relation is missing, otherwise say
"COMPLETE".

-Document-
{document}

-Extracted so far-
{extracted}
"""

RELATIONSHIP_EXTRACTION_PROMPT = """You are an embedded system developer skilled in domain specific customization. You are knowledgeable about general domain knowledge about RTOS, MCU, etc.

-Goal-
Given a technical document with a set of entities, identify the dependencies between the entities and return the output. For each relationships, extract the source, type and target. Return output in English as
(<source><|><relationship><|><target>). Use ## as the list delimiter. When finished, output <|COMPLETE|>

-Schema-
There are two types of entities: code entities and semantic entities. Code entities are code elements such as functions, structs, macros, enums, or header files in the C language, and pins, registers in the hardware datasheet. Semantic entities are semantic concepts such as tasks or system modules.

The relationship types include:
- USE: <code> -USE-> <code>: A code entity uses another code entity. For example, a function uses a data type, a function calls another function, a struct contains a member of another struct, etc.
- CONTAIN: <semantic> -CONTAIN-> <semantic>: A system module contains some submodules, a task decomposed into subtasks, etc.
- IMPLEMENT: <semantic> -IMPLEMENT-> <code>: A task is implemented by a function, a system module is implemented by a set of functions, etc.

#####################
-Example 1-
#####################
-Input Document-
esp_err_t esp_spp_init(esp_spp_mode_t mode)
This function is called to ...

-Entity Set-
(code<|>esp_spp_init<|>FUNCTION<|>This is a function called to initialize the SPP module.)##
(semantic<|>Initializing SPP module<|>TASK<|>This represents the process of initializing the SPP module.)##
#####################
output:
(esp_spp_init<|>USE<|>esp_spp_mode_t)##
(esp_spp_init<|>USE<|>esp_err_t)##
(Initializing SPP module<|>IMPLEMENT<|>esp_spp_init)##
<|COMPLETE|>

#####################
-Real Data-
#####################
-Input Document-
{document}
-Entity Set-
{entity_set}
#####################
output:
"""

QUERY_DECOMPOSITION_PROMPT = """You are an embedded system developer skilled in domain specific customization. You are knowledgeable about general domain knowledge about RTOS, MCU, etc.

-Goal-
Given a list of technical documents (each is an API Reference or Datasheet for a specific module) and a user query, first identify which documents may contain the answer to the query. Then, write a subquery for each candidate document to find the answer to the user query.

#####################
- Example -
Document List:
["stm32_api_reference", "DHT11_datasheet", "OLED_datasheet"]
User Query:
How to read the temperature of DHT11 sensor through GPIO?

- Output -
{{
    "candidate_documents": ["stm32_api_reference", "DHT11_datasheet"],
    "sub_queries": [
        {{
            "document": "stm32_api_reference",
            "sub_query": "How to read data from a GPIO pin?"
        }},
        {{
            "document": "DHT11_datasheet",
            "sub_query": "How to read the temperature data from the sensor?"
        }}
    ]
}}
#####################
-Real Data-
#####################
Document List:
{document_list}

User Query:
{user_query}

- Output -
"""

ANSWER_PROMPT = """You are an embedded system expert. Using ONLY the context below, answer the
user's question with specific, concise, step-by-step detail (exact function/parameter names where
available).

-Context-
{context}

-Question-
{question}

-Answer-
"""

RESPONSE_EVAL_PROMPT = """You are an expert in embedded system domain specific customization. You are knowledgeable about general domain knowledge about RTOS, MCU, etc.

-Goal-
Given a question with a reference answer, and two responses written by two people, your goal is to evaluate the responses and select the better one. You should read the criteria carefully and for each metric, decide which response is better or tie.

-Criteria-
- Specificity: Identify which answer is more specific, such as explicitly mentioning the exact functions, commands, or steps required, rather than providing a vague or general description.
- Conciseness: Determine which response directly addresses the question in a clear and focused manner, avoiding unnecessary or unrelated information.
- Completeness: Assess which answer provides a more comprehensive explanation, covering all the key points necessary to fully resolve the question.

-Output Format-
{{
    "title": "ResponseEvaluation",
    "type": "object",
    "properties": {{
        "specificity": {{"type": "string", "enum": ["A", "B", "Tie"]}},
        "conciseness": ...,
        "completeness": ...
    }}
}}

-Question-
{question}

-Reference Answer-
{reference_answer}

-Response A-
{response_a}

-Response B-
{response_b}
"""
