#!/usr/bin/env python
# universal_sql_qa.py  ——  通用 SQL 数据库问答系统（使用 LangChain Agent）
# 依赖：pip install openai gradio python-dotenv langchain langchain-openai langchain-community
# 支持：SQLite, MySQL, PostgreSQL 等数据库

import os, json, re, pathlib
from typing import Optional, List, Dict, Any
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit      # ★
from langchain.agents import initialize_agent                          # ★
from langchain.chains import create_sql_query_chain

# ---------- 配置类 ----------
class DatabaseConfig:
    """数据库配置类"""
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
    """问答系统配置类"""
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.0,
                 max_results: int = 5,
                 verbose: bool = False):
        self.model_name = model_name
        self.temperature = temperature
        self.max_results = max_results
        self.verbose = verbose

# ---------- 默认配置 ----------
DEFAULT_DB_CONFIG = DatabaseConfig(
    db_type="sqlite",
    db_path=os.path.join(os.path.dirname(__file__), "chembl_35.db")
)

DEFAULT_QA_CONFIG = QASystemConfig(
    model_name="gpt-4o-mini",
    temperature=0.0,
    verbose=False
)

# ---------- 载入环境变量 ----------
load_dotenv()

# ---------- 语言检测 ----------
def detect_language(text: str) -> str:
    """检测输入语言"""
    return "chinese" if re.search(r"[\u4e00-\u9fff]", text) else "english"

# ---------- 通用数据库问答系统类 ----------
class UniversalSQLQASystem:
    """通用 SQL 数据库问答系统（使用 Agent）"""
    
    def __init__(self, db_config: DatabaseConfig = None, qa_config: QASystemConfig = None):
        self.db_config = db_config or DEFAULT_DB_CONFIG
        self.qa_config = qa_config or DEFAULT_QA_CONFIG
        
        # 初始化数据库连接
        self.db = SQLDatabase.from_uri(self.db_config.db_uri)
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=self.qa_config.model_name,
            temperature=self.qa_config.temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 1）原来的 create_sql_query_chain 先留着
        self.write_query = create_sql_query_chain(self.llm, self.db)

        # 2）再加一个 Agent ——它能自己写 SQL、执行、解析错误
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)          # ★
        
        # 自定义 prompt prefix 用于生物医学数据库查询
        prompt_prefix = """
You are an expert in querying biomedical databases, especially ChEMBL.
Use SQL to answer questions.
If the question mentions 'clinical candidates', it means max_phase between 1 and 3.
If the question mentions 'approved drugs', it means max_phase = 4.
If the question mentions 'molecular weight less than 500', use full_mwt < 500.
If the question mentions 'small molecules', assume molfile IS NOT NULL.
"""
        
        self.agent = initialize_agent(
            tools=toolkit.get_tools(),
            llm=self.llm,
            agent_type="openai-tools",
            verbose=True,
            agent_kwargs={"prefix": prompt_prefix},
        )
    
    def pre_process_question(self, question: str) -> str | None:
        """识别常见句式并返回对应 SQL"""
        q = question.lower()
        if (
            "clinical candidate" in q and
            "approved drug" in q and
            ("molecular weight" in q or "less than 500" in q)
        ):
            return '''
                SELECT 
                  SUM(CASE WHEN md.max_phase BETWEEN 1 AND 3 THEN 1 ELSE 0 END) AS clinical_candidates,
                  SUM(CASE WHEN md.max_phase = 4 THEN 1 ELSE 0 END) AS approved_drugs
                FROM molecule_dictionary md
                JOIN compound_properties cp ON md.molregno = cp.molregno
                WHERE cp.full_mwt < 500
                AND md.therapeutic_flag = 1;
            '''
        return None
    
    def query_database(self, question: str, language: str = "english") -> str:
        """先让 Agent 试，如果它失败再 fallback 到原逻辑"""
        
        # 首先检查是否有预处理的SQL
        pre_sql = self.pre_process_question(question)
        if pre_sql:
            try:
                result = self.db.run(pre_sql)
                return self._format_result(result, language)
            except Exception as e:
                print(f"Pre-processed SQL failed: {e}")
                # 如果预处理失败，继续使用Agent
        
        try:
            # 捕获Agent的详细输出
            import io
            import sys
            
            # 重定向stdout来捕获Agent的输出
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout
            
            try:
                result = self.agent.run(question)
                # 获取捕获的输出
                agent_output = new_stdout.getvalue()
            finally:
                sys.stdout = old_stdout
            
            # 清理ANSI颜色代码
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            agent_output = ansi_escape.sub('', agent_output)
            result = ansi_escape.sub('', result)
            
            # 清理 markdown 格式
            if isinstance(result, str):
                # 移除 sql 代码块标记
                result = re.sub(r'```sql\s*', '', result)
                result = re.sub(r'```\s*$', '', result)
                # 移除其他可能的 markdown 格式
                result = re.sub(r'^\s*`\s*', '', result)
                result = re.sub(r'\s*`\s*$', '', result)
            
            # 提取关键信息并格式化输出
            if agent_output.strip():
                lines = agent_output.strip().split('\n')
                final_action_input = ""
                final_observation = ""
                final_answer = ""
                
                # 只保留最后一个Action Input、Observation和Final Answer
                for line in lines:
                    if line.startswith('Action Input:'):
                        # 只保留包含SQL查询的Action Input
                        if '"SELECT' in line:
                            final_action_input = line
                    elif line.startswith('Observation:'):
                        final_observation = line
                    elif line.startswith('Final Answer:'):
                        final_answer = line
                
                # 组合输出
                output_parts = []
                if final_action_input:
                    output_parts.append(final_action_input)
                if final_observation:
                    output_parts.append(final_observation)
                if final_answer:
                    output_parts.append(final_answer)
                
                if output_parts:
                    return "\n".join(output_parts)
            
            return result
            
        except Exception as e:
            print(f"Agent failed: {e}; fallback to chain...")
            # 原来的 self.query_chain.invoke() 继续保留
            try:
                # 使用 Chain 生成 SQL
                sql_query = self.write_query.invoke({"question": question})
                print(f"Generated SQL: {sql_query}")
                
                # 执行 SQL
                result = self.db.run(sql_query)
                
                # 格式化结果
                formatted_result = self._format_result(result, language)
                return formatted_result
                
            except Exception as chain_error:
                error_msg = f"❌ Query failed: {str(chain_error)}" if language == "english" else f"❌ 查询失败：{str(chain_error)}"
                print(f"Chain execution error: {chain_error}")
                return error_msg
    
    def _format_result(self, result: Any, language: str) -> str:
        """格式化查询结果"""
        if not result:
            return "No results found" if language == "english" else "无查询结果"
        
        # 如果结果是字符串，直接返回
        if isinstance(result, str):
            return result
        
        # 如果结果是列表
        if isinstance(result, list):
            if not result:
                return "No results found" if language == "english" else "无查询结果"
            
            # 检查是否有错误
            if "error" in result[0]:
                error_msg = f"❌ Error: {result[0]['error']}" if language == "english" else f"❌ 执行错误：{result[0]['error']}"
                return error_msg
            
            # COUNT 查询特殊处理
            if len(result) == 1 and next(iter(result[0])).lower().startswith("count"):
                num = next(iter(result[0].values()))
                return f"🔢 Count: {num}" if language == "english" else f"🔢 查询数量：{num}"
            
            # 普通结果
            header = (
                f"{len(result)} record(s) found (showing first {self.qa_config.max_results}):"
                if language == "english"
                else f"共 {len(result)} 条记录（仅展示前 {self.qa_config.max_results} 条）："
            )
            
            out = [header]
            for i, row in enumerate(result[:self.qa_config.max_results], 1):
                out.append(f"{i}. " + ", ".join(f"{k}: {str(v)[:50]}" for k, v in row.items()))
            return "\n".join(out)
        
        # 其他类型的结果
        return str(result)

# ---------- 创建默认系统实例 ----------
qa_system = UniversalSQLQASystem()

# ---------- Gradio 界面函数 ----------
def ask_question(question: str, history: list):
    """Gradio 界面函数"""
    language = detect_language(question)
    reply = qa_system.query_database(question, language)
    history.append([question, reply])
    return history, history

# ---------- Gradio 界面 ----------
with gr.Blocks(title="通用 SQL 数据库问答系统 (LangChain Agent)") as demo:
    gr.Markdown("# 通用 SQL 数据库问答系统 / Universal SQL Database Q&A\n💡 使用 LangChain Agent 的智能查询系统 / Intelligent query system using LangChain Agent")
    chatbot = gr.Chatbot(height=600)
    msg = gr.Textbox(
        label="你的问题 / Your Question", 
        placeholder="中文示例：查询用户表中的所有用户 | English example: Show all users from the users table"
    )
    submit = gr.Button("🚀 提交 / Submit")
    clear = gr.Button("🗑️ 清空对话 / Clear")
    state = gr.State([])

    submit.click(ask_question, inputs=[msg, state], outputs=[chatbot, state])
    msg.submit(ask_question, inputs=[msg, state], outputs=[chatbot, state])
    clear.click(lambda: ([], []), outputs=[chatbot, state])

if __name__ == "__main__":
    demo.launch(share=False, debug=True)
