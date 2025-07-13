#!/usr/bin/env python
# universal_sql_qa.py  ——  通用 SQL 数据库问答系统（使用 LangChain Agent）
# 依赖：pip install openai gradio python-dotenv langchain langchain-openai langchain-community
# 支持：SQLite, MySQL, PostgreSQL 等数据库

import os, json, re, pathlib, time
from typing import Optional, List, Dict, Any
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit      # ★
from langchain_community.agent_toolkits.sql.base import create_sql_agent  # ★ 修复导入
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
        self.agent  = create_sql_agent(toolkit=toolkit, llm=self.llm, verbose=True, handle_parsing_errors=True)  # ★
    
    def query_database(self, question: str, language: str = "english") -> str:
        """先让 Agent 试，如果它失败再 fallback 到原逻辑"""
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

# ---------- 创建数据库系统实例 ----------
# SQLite 数据库实例
sqlite_config = DatabaseConfig(
    db_type="sqlite",
    db_path=os.path.join(os.path.dirname(__file__), "chembl_35.db")
)
qa_system_sqlite = UniversalSQLQASystem(db_config=sqlite_config)

# 数据库选择器
def get_qa_system(db_choice: str):
    """根据选择返回对应的数据库系统"""
    return qa_system_sqlite  # 目前只支持 SQLite

# 数据库连接测试函数
def test_database_connection():
    """测试数据库连接和表结构"""
    print("\n" + "="*60)
    print("🔍 数据库连接测试 / Database Connection Test")
    print("="*60)
    
    # 测试 SQLite
    try:
        print("\n📊 测试 SQLite 数据库...")
        sqlite_test = qa_system_sqlite.db.run("SELECT name FROM sqlite_master WHERE type='table'")
        
        # 处理不同类型的查询结果
        if isinstance(sqlite_test, list) and len(sqlite_test) > 0:
            print(f"✅ SQLite 连接成功，找到 {len(sqlite_test)} 个表")
            print("📋 SQLite 表列表:")
            for i, table in enumerate(sqlite_test[:5], 1):  # 只显示前5个表
                if isinstance(table, dict) and 'name' in table:
                    print(f"   {i}. {table['name']}")
                elif isinstance(table, (list, tuple)) and len(table) > 0:
                    print(f"   {i}. {table[0]}")
            if len(sqlite_test) > 5:
                print(f"   ... 还有 {len(sqlite_test) - 5} 个表")
        elif isinstance(sqlite_test, str):
            print(f"✅ SQLite 连接成功，查询结果: {sqlite_test}")
        else:
            print("⚠️ SQLite: 查询结果格式异常")
            print(f"   结果类型: {type(sqlite_test)}")
            print(f"   结果内容: {sqlite_test}")
    except Exception as e:
        print(f"❌ SQLite 连接失败: {e}")
    
    print("="*60)

# 运行数据库连接测试
test_database_connection()

# 简单的数据库测试查询函数
def quick_database_test():
    """快速测试数据库是否有数据"""
    print("\n" + "="*60)
    print("🚀 快速数据库测试 / Quick Database Test")
    print("="*60)
    
    # 测试 SQLite
    print("\n📊 测试 SQLite 数据库...")
    try:
        # 检查数据库文件是否存在
        sqlite_file = os.path.join(os.path.dirname(__file__), "chembl_35.db")
        if os.path.exists(sqlite_file):
            file_size = os.path.getsize(sqlite_file) / (1024**3)  # GB
            print(f"✅ SQLite 数据库文件存在: {sqlite_file} ({file_size:.1f} GB)")
        else:
            print(f"❌ SQLite 数据库文件不存在: {sqlite_file}")
            print("💡 请确保 chembl_35.db 文件在程序同一目录下")
            return
        
        # 首先测试简单连接
        print("🔍 测试 SQLite 连接...")
        simple_test = qa_system_sqlite.db.run("SELECT 1 as test")
        print(f"✅ SQLite 连接正常，简单查询结果: {simple_test}")
        
        # 使用数据库内置方法获取表名
        print("🔍 使用内置方法获取表名...")
        try:
            table_names = qa_system_sqlite.db.get_table_names()
            print(f"✅ SQLite 内置方法找到 {len(table_names)} 个表: {table_names[:5]}")
            if len(table_names) > 5:
                print(f"   ... 还有 {len(table_names) - 5} 个表")
        except Exception as e:
            print(f"⚠️ 内置方法失败: {e}")
        
        # 尝试一个简单的查询
        sqlite_result = qa_system_sqlite.db.run("SELECT COUNT(*) as total_tables FROM sqlite_master WHERE type='table'")
        
        print(f"🔍 SQLite 查询结果类型: {type(sqlite_result)}")
        print(f"🔍 SQLite 查询结果内容: {sqlite_result}")
        
        if isinstance(sqlite_result, list) and len(sqlite_result) > 0:
            table_count = None
            if isinstance(sqlite_result[0], dict) and 'total_tables' in sqlite_result[0]:
                table_count = sqlite_result[0]['total_tables']
            elif isinstance(sqlite_result[0], (list, tuple)) and len(sqlite_result[0]) > 0:
                table_count = sqlite_result[0][0]
            
            if table_count is not None:
                print(f"✅ SQLite: 找到 {table_count} 个表")
                
                if table_count > 0:
                    # 尝试查询第一个表的数据
                    first_table = qa_system_sqlite.db.run("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                    if isinstance(first_table, list) and len(first_table) > 0:
                        table_name = None
                        if isinstance(first_table[0], dict) and 'name' in first_table[0]:
                            table_name = first_table[0]['name']
                        elif isinstance(first_table[0], (list, tuple)) and len(first_table[0]) > 0:
                            table_name = str(first_table[0][0])
                        
                        if table_name:
                            sample_data = qa_system_sqlite.db.run(f"SELECT * FROM {table_name} LIMIT 1")
                            if isinstance(sample_data, list) and len(sample_data) > 0:
                                print(f"✅ SQLite: 表 '{table_name}' 有数据")
                            else:
                                print(f"⚠️ SQLite: 表 '{table_name}' 是空的")
        else:
            print("❌ SQLite: 无法获取表信息")
    except Exception as e:
        print(f"❌ SQLite 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*60)

# 运行快速测试
quick_database_test()

# ---------- Gradio 界面函数 ----------
# 返回：消息记录、SQL语句、表格结果
def ask_question(question: str, db_choice: str, history: list):
    language = detect_language(question)
    qa_system = get_qa_system(db_choice)
    
    # 导入 pandas
    import pandas as pd
    
    try:
        start_time = time.time()
        reply = qa_system.query_database(question, language)
        end_time = time.time()
        duration = end_time - start_time

        # SQL语句提取（改进的提取逻辑）
        sql_text = "(未提取到 SQL)"
        
        # 尝试多种方式提取 SQL
        sql_patterns = [
            r'(?i)SELECT.+?FROM.+?(?:;|$)',
            r'(?i)SELECT.+?FROM.+',
            r'```sql\s*(SELECT.+?)\s*```',
            r'SQLQuery:\s*(SELECT.+?)(?:;|$)',
            r'SQLQuery:\s*(.+)'
        ]
        
        for pattern in sql_patterns:
            sql_match = re.search(pattern, reply, re.DOTALL)
            if sql_match:
                sql_text = sql_match.group(1) if 'group' in sql_match.groupdict() else sql_match.group(0)
                # 清理 SQL 语句
                sql_text = sql_text.replace('SQLQuery:', '').strip()
                # 移除可能的 markdown 格式
                sql_text = re.sub(r'```sql\s*', '', sql_text)
                sql_text = re.sub(r'```\s*$', '', sql_text)
                sql_text = sql_text.strip()
                break

        # 格式化表格（尝试从 db.run 的结果结构化，如果无法结构化就用空表）
        df = pd.DataFrame()  # 默认空 DataFrame
        try:
            if sql_text != "(未提取到 SQL)":
                # 再次执行 SQL 结构化返回
                df_result = qa_system.db.run(sql_text)
                if df_result:
                    df = pd.DataFrame(df_result)
        except Exception as sql_error:
            print(f"SQL 执行错误: {sql_error}")
            df = pd.DataFrame()

        # 添加到聊天记录
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"[{db_choice}] ✅ 查询成功，用时 {duration:.2f} 秒\n\n🔍 SQL:\n```sql\n{sql_text}\n```\n\n📊 结果:\n{reply}"})

    except Exception as e:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"❌ 数据库错误: {str(e)}"})
        df = pd.DataFrame()

    return history, history, df

# ---------- Gradio 界面 ----------
with gr.Blocks(title="通用 SQL 数据库问答系统 (LangChain Agent)") as demo:
    gr.Markdown("# 通用 SQL 数据库问答系统 / Universal SQL Database Q&A\n💡 使用 LangChain Agent 的智能查询系统 / Intelligent query system using LangChain Agent")
    
    # 数据库选择器（目前只有 SQLite）
    db_choice = gr.Dropdown(
        choices=["SQLite"],
        value="SQLite",
        label="选择数据库 / Select Database",
        info="SQLite: 本地文件数据库"
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=600, type="messages")
            msg = gr.Textbox(
                label="你的问题 / Your Question", 
                placeholder="中文示例：查询用户表中的所有用户 | English example: Show all users from the users table"
            )
            with gr.Row():
                submit = gr.Button("🚀 提交 / Submit")
                clear = gr.Button("🗑️ 清空对话 / Clear")
            state = gr.State([])
        
        with gr.Column(scale=1):
            gr.Markdown("## 📊 查询结果表格 / Query Results Table")
            result_table = gr.Dataframe(
                headers=["结果"],
                datatype=["str"],
                label="查询结果 / Query Results"
            )

    submit.click(ask_question, inputs=[msg, db_choice, state], outputs=[chatbot, state, result_table])
    msg.submit(ask_question, inputs=[msg, db_choice, state], outputs=[chatbot, state, result_table])
    clear.click(lambda: ([], [], gr.DataFrame()), outputs=[chatbot, state, result_table])

if __name__ == "__main__":
    demo.launch(share=False, debug=True)
