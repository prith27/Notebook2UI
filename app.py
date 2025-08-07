import streamlit as st
import nbformat
from openai import OpenAI
import os
from dotenv import load_dotenv
import traceback
import streamlit.components.v1 as components
from tempfile import NamedTemporaryFile
import subprocess
import re
import ast
import importlib.util

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def strip_code_fence(generated_code: str) -> str:
    """
    Removes markdown-style code fences from GPT responses.
    Handles ```python, ```, and '''.
    """
    pattern = r"^\s*```(?:python)?\s*\n([\s\S]*?)\n```$|^\s*'''\s*\n([\s\S]*?)\n'''$"
    match = re.match(pattern, generated_code.strip())
    
    if match:
        return match.group(1) or match.group(2)
    return generated_code.strip()


def extract_imports_from_code(code: str) -> set:
    """
    Extract imported modules from Python code.
    """
    imports = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except:
        # Fallback to regex if AST parsing fails
        import_pattern = r'(?:from\s+(\w+)|import\s+(\w+))'
        matches = re.findall(import_pattern, code)
        for match in matches:
            imports.add(match[0] if match[0] else match[1])
    
    return imports


def generate_requirements_txt(imports: set) -> str:
    """
    Generate requirements.txt content based on imports.
    """
    # Common package mappings
    package_mappings = {
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'skimage': 'scikit-image',
        'bs4': 'beautifulsoup4',
        'yaml': 'PyYAML',
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'scipy': 'scipy',
        'requests': 'requests',
        'flask': 'Flask',
        'fastapi': 'fastapi',
        'tensorflow': 'tensorflow',
        'torch': 'torch',
        'transformers': 'transformers',
        'openai': 'openai',
        'langchain': 'langchain',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'joblib': 'joblib',
        'pickle': '',  # built-in
        'json': '',   # built-in
        'os': '',     # built-in
        're': '',     # built-in
        'sys': '',    # built-in
        'datetime': '', # built-in
        'time': '',   # built-in
        'math': '',   # built-in
        'random': '', # built-in
        'collections': '', # built-in
        'itertools': '', # built-in
        'functools': '', # built-in
        'pathlib': '', # built-in
        'tempfile': '', # built-in
        'traceback': '', # built-in
        'subprocess': '', # built-in
        'dotenv': 'python-dotenv'
    }
    
    requirements = []
    for imp in imports:
        if imp in package_mappings:
            if package_mappings[imp]:  # Not a built-in module
                requirements.append(package_mappings[imp])
        elif imp not in ['__future__', 'typing']:  # Skip special modules
            requirements.append(imp)
    
    # Always include streamlit if not present
    if 'streamlit' not in requirements:
        requirements.append('streamlit')
    
    return '\n'.join(sorted(set(requirements)))


# Enhanced Streamlit UI
st.set_page_config(
    page_title="Jupyter ➜ Streamlit Converter", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📓"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 2rem;
}

.info-card {
    background-color: #f0f2f6;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #4ecdc4;
    margin: 1rem 0;
    color: #000000; /* or simply: color: black; */
}

.success-card {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
    color:#000000
}

.code-section {
    background-color: #000000;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.stButton > button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<h1 class="main-header">📓 Jupyter to Streamlit Converter</h1>', unsafe_allow_html=True)

# Sidebar with instructions
with st.sidebar:
    st.header("📋 Instructions")
    
    st.markdown("""
    ### How to use this app:
    1. **Upload** your Jupyter notebook (.ipynb file)
    2. **Wait** for AI analysis and code generation
    3. **Review** the generated Streamlit app code
    4. **Download** the requirements.txt file
    5. **Copy** the generated code or download it
    6. **Run** your new Streamlit app!
    
    ### 🚀 How to run the generated app:
    ```bash
    # Install requirements
    pip install -r requirements.txt
    
    # Run the app
    streamlit run your_app.py
    ```
    
    ### ⚠️ Note:
    - Make sure you have your OpenAI API key in .env file
    - Generated code may need minor adjustments
    - Test the app before deploying to production
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>🎯 What this app does:</h3>
        <ul>
            <li>Converts Jupyter notebooks to interactive Streamlit apps</li>
            <li>Automatically generates UI elements for data input/output</li>
            <li>Creates requirements.txt for easy deployment</li>
            <li>Provides setup instructions and preview functionality</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>📊 Supported Features:</h3>
        <ul>
            <li>Data visualization</li>
            <li>Machine learning models</li>
            <li>File upload/download</li>
            <li>Interactive widgets</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# File uploader with enhanced styling
st.markdown("### 📁 Upload Your Notebook")
uploaded_file = st.file_uploader(
    "Choose a Jupyter Notebook file", 
    type=["ipynb"],
    help="Upload your .ipynb file to convert it to a Streamlit app"
)

if uploaded_file is not None:
    try:
        # Parse notebook
        notebook = nbformat.read(uploaded_file, as_version=4)
        code_cells = []
        markdown_cells = []

        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                code_cells.append(cell['source'])
            elif cell['cell_type'] == 'markdown':
                markdown_cells.append(cell['source'])

        notebook_content = "\n\n".join(markdown_cells + code_cells)

        # Show notebook statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Total Cells", len(notebook['cells']))
        with col2:
            st.metric("💻 Code Cells", len(code_cells))
        with col3:
            st.metric("📄 Markdown Cells", len(markdown_cells))
        with col4:
            st.metric("📊 File Size", f"{len(notebook_content)} chars")

        with st.spinner("🧠 Analyzing notebook with AI..."):
            ### STEP 1: Summarize ###
            summary_prompt = f"""You are an expert ML assistant. Read the following code and markdown from a Jupyter notebook. 
Summarize what the notebook is trying to do in 3-4 lines for a technical reader.

Content:
{notebook_content}
"""
            summary_response = client.chat.completions.create(
                model="gpt-4.1-nano-2025-04-14",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.4
            )
            summary = summary_response.choices[0].message.content

            ### STEP 2: Generate Streamlit Code ###
            streamlit_prompt = f"""You are an expert Python and Streamlit developer. Convert the following notebook code and markdown into a Streamlit app.
Wrap input/output with UI elements like file uploaders, sliders, charts, etc. Make sure to allow user to upload the same type of data if required.
Include proper error handling and user-friendly interface elements.

Notebook content:
{notebook_content}

Output only the code. Do not explain anything.
"""
            streamlit_response = client.chat.completions.create(
                model="gpt-4.1-nano-2025-04-14",
                messages=[{"role": "user", "content": streamlit_prompt}],
                temperature=0.4
            )
            streamlit_code_raw = streamlit_response.choices[0].message.content
            streamlit_code = strip_code_fence(streamlit_code_raw)

        # Success message
        st.markdown("""
        <div class="success-card">
            <h3>✅ Notebook successfully processed!</h3>
            <p>Your Jupyter notebook has been converted to a Streamlit app. Review the code below and download the requirements.</p>
        </div>
        """, unsafe_allow_html=True)

        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "💻 Generated Code", "📦 Requirements", "🚀 How to Run"])

        with tab1:
            st.markdown("### 📌 Notebook Summary")
            st.markdown(f"""
            <div class="code-section">
                <p><strong>Analysis:</strong> {summary.strip()}</p>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.markdown("### 💻 Generated Streamlit Code")
            st.code(streamlit_code, language="python")
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Python File",
                    data=streamlit_code,
                    file_name="streamlit_app.py",
                    mime="text/x-python",
                    use_container_width=True
                )
            
            with col2:
                # Copy button with JavaScript
                if st.button("📋 Copy to Clipboard", use_container_width=True):
                    st.write("Code copied! (Use Ctrl+C to copy manually if needed)")

        with tab3:
            st.markdown("### 📦 Requirements & Dependencies")
            
            # Extract imports and generate requirements
            imports = extract_imports_from_code(streamlit_code)
            requirements_content = generate_requirements_txt(imports)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Detected Imports:**")
                if imports:
                    for imp in sorted(imports):
                        st.write(f"• {imp}")
                else:
                    st.write("No imports detected")
            
            with col2:
                st.markdown("**requirements.txt:**")
                st.code(requirements_content, language="text")
                
                st.download_button(
                    label="📥 Download requirements.txt",
                    data=requirements_content,
                    file_name="requirements.txt",
                    mime="text/plain",
                    use_container_width=True
                )

        with tab4:
            st.markdown("### 🚀 How to Run Your App")
            
            st.markdown("""
            <div class="info-card">
                <h4>Step-by-step instructions:</h4>
            </div>
            """, unsafe_allow_html=True)
            
            steps = [
                ("1️⃣ **Save the files**", "Download both the Python file and requirements.txt to the same folder"),
                ("2️⃣ **Install dependencies**", "Run `pip install -r requirements.txt` in your terminal"),
                ("3️⃣ **Run the app**", "Execute `streamlit run streamlit_app.py` in your terminal"),
                ("4️⃣ **Open in browser**", "Your app will open automatically at `http://localhost:8501`")
            ]
            
            for step, description in steps:
                st.markdown(f"""
                <div style="margin: 1rem 0; padding: 1rem; background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid #007bff; color: #000000;">
                    <strong>{step}</strong><br>
                    {description}
                </div>
                """, unsafe_allow_html=True)
            
            st.code("""
# Terminal commands:
cd /path/to/your/app/folder
pip install -r requirements.txt
streamlit run streamlit_app.py
            """, language="bash")
            
            st.warning("⚠️ **Note:** Make sure you have Python 3.7+ installed and that all dependencies are compatible with your system.")

        # Optional preview section (commented out as it requires local setup)
        """
        ### 🔍 Preview (Local Development Only)
        if st.button("🚀 Try to Preview App Locally"):
            try:
                with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
                    temp_file.write(streamlit_code)
                    temp_path = temp_file.name

                process = subprocess.Popen(["streamlit", "run", temp_path, "--server.port=8502"])
                st.success("✅ Preview started on port 8502!")
                st.info("📱 Open http://localhost:8502 in a new browser tab to see your app.")
                
            except Exception as e:
                st.error("❌ Couldn't start preview. Make sure Streamlit is installed locally.")
                st.exception(e)
        """

    except Exception as e:
        st.error("❌ Failed to process notebook.")
        st.markdown("""
        <div style="background-color: #f8d7da; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc3545;">
            <h4>Error Details:</h4>
        </div>
        """, unsafe_allow_html=True)
        st.code(traceback.format_exc())
        
        st.markdown("""
        ### 🔧 Troubleshooting Tips:
        - Make sure your notebook file is valid
        - Check if all cells can be executed
        - Verify that your OpenAI API key is configured
        - Try with a simpler notebook first
        """)

else:
    # Landing section when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <h2>🚀 Ready to convert your Jupyter notebook?</h2>
        <p style="font-size: 1.2rem; color: #666;">
            Upload your .ipynb file above to get started!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show example of what the tool can do
    with st.expander("📖 See Example Output"):
        st.code("""
# Example of generated Streamlit code:
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Data Analysis App")

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    
    if st.button("Generate Plot"):
        fig, ax = plt.subplots()
        df.hist(ax=ax)
        st.pyplot(fig)
        """, language="python")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Built with ❤️ using Streamlit and OpenAI GPT-4</p>
</div>
""", unsafe_allow_html=True)
