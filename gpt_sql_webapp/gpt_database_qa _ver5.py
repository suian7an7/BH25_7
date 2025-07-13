#!/usr/bin/env python
# universal_sql_qa.py  ——  Universal SQL Database Q&A System (using LangChain Agent)
# Dependencies: pip install openai gradio python-dotenv langchain langchain-openai langchain-community
# Supports: SQLite, MySQL, PostgreSQL and other databases

import os, json, re, pathlib
from typing import Optional, List, Dict, Any
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit      # ★
from langchain.agents import initialize_agent                          # ★
from langchain.chains import create_sql_query_chain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentType

# ---------- Configuration Classes ----------
class DatabaseConfig:
    """Database configuration class"""
    def __init__(self, db_type: str = "sqlite", db_path: str = None, db_uri: str = None):
        self.db_type = db_type.lower()
        self.db_path = db_path
        self.db_uri = db_uri
        
        if not db_uri:
            if self.db_type == "sqlite":
                self.db_uri = f"sqlite:///{db_path}"
            elif self.db_type == "mysql":
                self.db_uri = f"mysql+pymysql://{db_path}"
            elif self.db_type == "postgresql":
                self.db_uri = f"postgresql://{db_path}"
            else:
                raise ValueError(f"Unsupported database type: {db_type}")

class QASystemConfig:
    """Q&A system configuration class"""
    def __init__(self, 
                 model_name: str = "gpt-4o",
                 temperature: float = 0.0,
                 max_results: int = 5,
                 verbose: bool = False):
        self.model_name = model_name
        self.temperature = temperature
        self.max_results = max_results
        self.verbose = verbose

# ---------- Default Configuration ----------
DEFAULT_DB_CONFIG = DatabaseConfig(
    db_type="sqlite",
    db_path=os.path.join(os.path.dirname(__file__), "chembl_35.db")
)

DEFAULT_QA_CONFIG = QASystemConfig(
    model_name="gpt-4o",
    temperature=0.0,
    verbose=False
)

# ---------- Load Environment Variables ----------
load_dotenv()

# ---------- Universal SQL Database Q&A System Class ----------
class UniversalSQLQASystem:
    """Universal SQL Database Q&A System (using Agent)"""
    
    def __init__(self, db_config: DatabaseConfig = None, qa_config: QASystemConfig = None):
        self.db_config = db_config or DEFAULT_DB_CONFIG
        self.qa_config = qa_config or DEFAULT_QA_CONFIG
        
        # Initialize database connection
        self.db = SQLDatabase.from_uri(self.db_config.db_uri)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.qa_config.model_name,
            temperature=self.qa_config.temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Generate system prompt from database schema
        self.system_prompt = self.generate_system_prompt()
        
        # Create prompt template with system prompt injection
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        
        # Create LLM chain with prompt template
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        
        # Keep original create_sql_query_chain
        self.write_query = create_sql_query_chain(self.llm, self.db)

        # Add memory for conversational agent (only keep last 2 rounds)
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=3  # 只保留最近 2 轮上下文
        )

        # Add Agent - it can write SQL, execute, and parse errors
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)          # ★
        self.agent  = initialize_agent(
            toolkit.get_tools(), self.llm, 
            agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True         
        )
    
    def generate_system_prompt(self) -> str:
        """Generate system prompt from database schema and semantic mapping file"""
        try:
            table_info = self.db.get_table_info()
            if isinstance(table_info, str):
                table_descriptions = []
                table_blocks = table_info.split('CREATE TABLE')
                for block in table_blocks[1:]:
                    lines = block.strip().split('\n')
                    if lines:
                        table_name_line = lines[0]
                        table_name = table_name_line.split('(')[0].strip().strip('"').strip("'")
                        columns = []
                        for line in lines[1:]:
                            line = line.strip()
                            if line and not line.startswith('PRIMARY KEY') and not line.startswith('FOREIGN KEY') and not line.startswith(')'):
                                column_match = re.match(r'^\s*(\w+)', line)
                                if column_match:
                                    columns.append(column_match.group(1))
                        if columns:
                            column_list = ", ".join(columns)
                            table_descriptions.append(f"- {table_name}: {column_list}")
                schema_hint = "\n".join(table_descriptions) if table_descriptions else "No table information available"
            else:
                schema_hint = "Database schema information not available"
        except Exception as e:
            print(f"Error generating system prompt: {e}")
            schema_hint = "Database schema information not available"

        # 🔽 Step 1: Determine semantic mapping file name
        db_base_name = pathlib.Path(self.db_config.db_path).stem.lower()  # e.g., 'chembl_35'
        mapping_file = f"semantic_mapping_{db_base_name}.txt"            # e.g., 'semantic_mapping_chembl_35.txt'

        # 🔽 Step 2: Load semantic mapping content
        semantic_hint = ""
        try:
            mapping_path = os.path.join(os.path.dirname(__file__), mapping_file)
            with open(mapping_path, "r", encoding="utf-8") as f:
                semantic_hint = f.read()
        except Exception as e:
            semantic_hint = "# (No semantic mapping loaded. Proceeding without external domain rules.)"
            print(f"Warning: Could not load {mapping_file}: {e}")

        # 🔽 Step 3: Combine into system prompt
        return (
            "You are an expert SQL assistant.\n"
            "The following tables and their columns are available in the database:\n\n"
            f"{schema_hint}\n\n"
            f"{semantic_hint}\n\n"
            "When answering a question, ONLY output the following three sections, and nothing else:\n"
            "1. SQL Query: (the SQL statement you would use)\n"
            "2. Query Result: (the raw result of the SQL query)\n"
            "3. Final Answer: (a concise natural language answer to the user's question, if applicable)\n"
            "Do NOT output your reasoning process, thoughts, step-by-step actions, or any intermediate steps.\n"
            "Do NOT output Action, Action Input, Observation, or Thought.\n"
            "Avoid using markdown formatting like ```sql.\n"
            "Always return only the three sections above, in order.\n"
            "If you are not sure or do not have enough information to answer, say so and do not attempt to guess.\n"
            "Join tables on obvious keys such as molregno if necessary."
        )
    
    def clean_sql_text(self, text: str) -> str:
        """Remove markdown code blocks from SQL text"""
        # Remove markdown code block wrappers
        text = text.strip()
        if text.startswith("```sql"):
            text = text[6:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
    
    def is_valid_sql(self, text: str) -> bool:
        """Check if text is valid SQL statement"""
        if not text:
            return False
        
        # Clean the text first
        cleaned_text = self.clean_sql_text(text)
        
        # Check if it starts with SQL keywords
        sql_keywords = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']
        return any(cleaned_text.upper().startswith(keyword) for keyword in sql_keywords)
    
    def query_database(self, question: str, chat_history=None) -> str:
        try:
            # 传递 chat_history 给 agent
            agent_output = self.agent.run(input=question, chat_history=chat_history or [])
            # Use regex to extract only the three sections
            import re
            sections = {}
            # Extract SQL Query
            sql_match = re.search(r'SQL Query\s*[:：]\s*(.*?)(?=\n\s*(Query Result|Final Answer|$))', agent_output, re.DOTALL | re.IGNORECASE)
            if sql_match:
                sections['SQL Query'] = sql_match.group(1).strip()
            # Extract Query Result
            result_match = re.search(r'Query Result\s*[:：]\s*(.*?)(?=\n\s*(Final Answer|SQL Query|$))', agent_output, re.DOTALL | re.IGNORECASE)
            if result_match:
                sections['Query Result'] = result_match.group(1).strip()
            # Extract Final Answer
            answer_match = re.search(r'Final Answer\s*[:：]\s*(.*)', agent_output, re.DOTALL | re.IGNORECASE)
            if answer_match:
                sections['Final Answer'] = answer_match.group(1).strip()
            # Build output
            output = []
            if 'SQL Query' in sections:
                output.append(f"**SQL Query:**\n{sections['SQL Query']}")
            if 'Query Result' in sections:
                output.append(f"**Query Result:**\n{sections['Query Result']}")
            if 'Final Answer' in sections:
                output.append(f"**Final Answer:**\n{sections['Final Answer']}")
            # If nothing matched, fallback to original output
            if not output:
                output = [agent_output]
            return "\n\n".join(output)
        except Exception as e:
            if "Agent stopped" in str(e) or "iteration limit" in str(e).lower():
                return "⚠️ The agent could not complete the query in time. Please simplify or rephrase your question."
            if "output parsing error" in str(e).lower():
                return f"⚠️ Agent could not parse the output. Here is the LLM's response:\n\n{str(e)}"
            error_msg = f"❌ Query failed: {str(e)}"
            print(f"Query execution error: {e}")
            return error_msg
    
    def _format_result(self, result: Any) -> str:
        """Format query results"""
        if not result:
            return "No results found"
        
        # If result is string, return directly
        if isinstance(result, str):
            return result
        
        # If result is list
        if isinstance(result, list):
            if not result:
                return "No results found"
            
            # Check for errors
            if "error" in result[0]:
                error_msg = f"❌ Error: {result[0]['error']}"
                return error_msg
            
            # COUNT query special handling
            if len(result) == 1 and next(iter(result[0])).lower().startswith("count"):
                num = next(iter(result[0].values()))
                return f"🔢 Count: {num}"
            
            # Regular results
            header = f"{len(result)} record(s) found (showing first {self.qa_config.max_results}):"
            
            out = [header]
            for i, row in enumerate(result[:self.qa_config.max_results], 1):
                out.append(f"{i}. " + ", ".join(f"{k}: {str(v)[:50]}" for k, v in row.items()))
            return "\n".join(out)
        
        # Other types of results
        return str(result)

# ---------- Create Default System Instance ----------
qa_system = UniversalSQLQASystem()

# ---------- Gradio Interface Functions ----------
def ask_question(question: str, history: list):
    """Gradio Interface Function"""
    reply = qa_system.query_database(question, chat_history=history)
    history.append([question, reply])
    return history, history

# ---------- Gradio Interface ----------
with gr.Blocks(title="Universal SQL Database Q&A System (LangChain Agent)") as demo:
    gr.Markdown("# Universal SQL Database Q&A System\n💡 Intelligent query system using LangChain Agent")
    chatbot = gr.Chatbot(height=600)
    msg = gr.Textbox(
        label="Your Question", 
        placeholder="Example: Show all users from the users table"
    )
    submit = gr.Button("🚀 Submit")
    clear = gr.Button("🗑️ Clear Chat")
    state = gr.State([])

    submit.click(ask_question, inputs=[msg, state], outputs=[chatbot, state])
    msg.submit(ask_question, inputs=[msg, state], outputs=[chatbot, state])
    clear.click(lambda: ([], []), outputs=[chatbot, state])

if __name__ == "__main__":
    demo.launch(share=False, debug=True)
