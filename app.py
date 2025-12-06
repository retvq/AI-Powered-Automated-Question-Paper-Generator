import gradio as gr
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Model Setup using HF Inference Client ---
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    print("⚠️ Warning: HF_TOKEN not set. The app may not work properly.")

# Use InferenceClient directly instead of LangChain wrapper
client = InferenceClient(token=HF_TOKEN)

# --- 2. The Core Logic ---
def generate_question_paper(
    pdf_files, 
    mcq_difficulty, mcq_count,
    short_difficulty, short_count,
    long_difficulty, long_count,
    num_sets,
    progress=gr.Progress()
):
    # Add timeout protection
    import time
    start_time = time.time()
    
    if not pdf_files or len(pdf_files) == 0:
        return "❌ Please upload at least one PDF file."
    
    if len(pdf_files) > 5:
        return "❌ Error: Maximum 5 PDF files allowed."
    
    if not HF_TOKEN:
        return "❌ Error: HF_TOKEN not configured. Please add your Hugging Face token in Space Settings > Repository secrets."
    
    total_questions = mcq_count + short_count + long_count
    if total_questions == 0:
        return "❌ Please specify at least one question."
    
    try:
        # A. Load all PDFs
        progress(0, desc=f"📄 PDF file(s) uploaded, accessing {len(pdf_files)} file(s)...")
        all_pages = []
        
        for idx, pdf_file in enumerate(pdf_files):
            current_progress = 0.05 + (idx * 0.1 / len(pdf_files))
            progress(current_progress, 
                    desc=f"📂 Accessing PDF {idx + 1}/{len(pdf_files)}: {pdf_file.name.split('/')[-1][:30]}...")
            loader = PyPDFLoader(pdf_file.name)
            pages = loader.load()
            
            if not pages:
                return f"❌ Error: Could not extract text from {pdf_file.name}. Please ensure it's a valid PDF with text content."
            
            all_pages.extend(pages)
        
        progress(0.15, desc=f"✅ PDF loaded successfully! Extracted {len(all_pages)} pages from {len(pdf_files)} file(s)")
        
        # B. Split Text
        progress(0.20, desc="📝 Extracting text content from PDFs...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(all_pages)
        progress(0.30, desc=f"✅ Text extracted successfully! Created {len(chunks)} text chunks, preparing embeddings...")
        
        # C. Vector Store (FAISS)
        progress(0.35, desc="🧠 Generating embeddings for content understanding...")
        embeddings = FastEmbedEmbeddings()
        progress(0.40, desc="🧠 Creating knowledge base from embeddings...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        progress(0.50, desc="✅ Knowledge base created successfully! Analyzing content for key concepts...")
        
        # D. Retrieve Context (more chunks for multiple PDFs)
        progress(0.55, desc="🔍 Identifying key concepts and topics from content...")
        retriever = vector_store.as_retriever(search_kwargs={"k": min(10, len(chunks))})
        context_docs = retriever.invoke("Key concepts, definitions, and important topics")
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        progress(0.60, desc=f"✅ Analysis complete! Found {len(context_docs)} key sections. Activating AI model...")
        
        # E. Generate all sets
        all_outputs = []
        
        for set_num in range(1, num_sets + 1):
            progress(0.65 + (set_num - 1) * 0.30 / num_sets, 
                    desc=f"🤖 AI Model activated! Preparing to generate Set {set_num}/{num_sets}...")
            
            # Create Prompt for this set
            sections = []
            answer_key_instructions = []
            
            if mcq_count > 0:
                sections.append(f"""Section A: Multiple Choice Questions (MCQs) - {mcq_count} questions
Difficulty: {mcq_difficulty}
Create {mcq_count} MCQs with 4 options each (A, B, C, D). Mark the correct answer clearly.""")
                answer_key_instructions.append("MCQ Answer Key")
            
            if short_count > 0:
                sections.append(f"""Section B: Short Answer Questions - {short_count} questions
Difficulty: {short_difficulty}
Create {short_count} short answer questions (2-3 marks each, expected answer: 2-3 sentences).""")
            
            if long_count > 0:
                sections.append(f"""Section C: Long Answer/Essay Questions - {long_count} questions
Difficulty: {long_difficulty}
Create {long_count} long answer questions (5-10 marks each, expected answer: detailed explanation).""")
            
            sections_text = "\n\n".join(sections)
            answer_key_text = "\n".join([f"- {key}" for key in answer_key_instructions])
            
            prompt = f"""You are an expert academic examiner. Create a formal Question Paper based ONLY on the context provided below.

CONTEXT:
{context_text}

INSTRUCTIONS:
Create Question Paper Set {set_num} of {num_sets}

{sections_text}

FORMAT REQUIREMENTS:
- Start with "QUESTION PAPER - SET {set_num}"
- Include proper section headers
- Number all questions sequentially within each section
- For MCQs: Provide 4 options (A, B, C, D)
- At the end, provide:
{answer_key_text}

Do not output conversational text. Output ONLY the exam paper in a well-formatted structure."""
            
            progress(0.70 + (set_num - 1) * 0.30 / num_sets, 
                    desc=f"✍️ Generating Question Paper Set {set_num}/{num_sets}... 0%")
            
            # F. Generate using chat completion
            messages = [{"role": "user", "content": prompt}]
            
            response = ""
            token_count = 0
            max_tokens = 2500  # Increased for longer papers
            last_update_time = time.time()
            
            try:
                for message in client.chat_completion(
                    messages=messages,
                    model="meta-llama/Llama-3.2-3B-Instruct",
                    max_tokens=max_tokens,
                    temperature=0.7,
                    stream=True,
                ):
                    # Check total timeout
                    if time.time() - start_time > 300:  # 5 minute total timeout
                        return f"⏱️ Request timeout. Please try with:\n- Fewer PDF files\n- Fewer questions\n- Fewer sets\n\nPartial output:\n{response}"
                    
                    if hasattr(message, 'choices') and len(message.choices) > 0:
                        if hasattr(message.choices[0], 'delta') and hasattr(message.choices[0].delta, 'content'):
                            response += message.choices[0].delta.content or ""
                            token_count += 1
                            
                            # Update progress every 50 tokens to reduce overhead
                            if token_count % 50 == 0 or time.time() - last_update_time > 2:
                                # Calculate progress within this set (70-95% range divided by number of sets)
                                set_start = 0.70 + (set_num - 1) * 0.30 / num_sets
                                set_range = 0.25 / num_sets  # 25% of total progress for generation
                                generation_progress = min((token_count / max_tokens), 1.0)
                                current_progress = set_start + (generation_progress * set_range)
                                percentage = int(generation_progress * 100)
                                
                                # Update with dynamic percentage
                                progress(current_progress, 
                                        desc=f"✍️ Generating Question Paper Set {set_num}/{num_sets}... {percentage}%")
                                last_update_time = time.time()
            
            except Exception as e:
                if response:
                    return f"⚠️ Generation interrupted: {str(e)}\n\nPartial output for Set {set_num}:\n{response}"
                else:
                    raise e
            
            progress(0.70 + set_num * 0.30 / num_sets, 
                    desc=f"✅ Set {set_num}/{num_sets} generated successfully!")
            
            all_outputs.append(response)
        
        progress(1.0, desc=f"✅ All {num_sets} Question Paper(s) Generated Successfully! 🎉")
        
        # Combine all sets
        final_output = "\n\n" + "="*80 + "\n\n".join(all_outputs)
        return final_output

    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease check:\n1. PDFs are valid and contain text\n2. HF_TOKEN is correctly set in Space secrets\n3. Try again or contact support"

# --- 3. The UI ---
with gr.Blocks(title="AI Question Paper Generator") as demo:
    gr.Markdown("# 📄 AI Question Paper Generator Pro")
    gr.Markdown("Powered by **Fine-Tuned Llama 3.2 3B**")
    gr.Markdown("⚡ Fast • 🎯 Accurate • 📚 Multi-PDF Support • 🎲 Multiple Sets")
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="📄 Upload Study Materials (PDF) - Max 5 files",
                file_types=[".pdf"],
                file_count="multiple"
            )
            
            gr.Markdown("### 🎲 Number of Question Paper Sets")
            num_sets = gr.Slider(
                1, 3, value=1, step=1,
                label="📋 Generate multiple unique sets"
            )
            
            gr.Markdown("### 📝 Section A: Multiple Choice Questions")
            with gr.Group():
                mcq_difficulty = gr.Radio(
                    ["Easy", "Medium", "Hard"], 
                    label="🎚️ MCQ Difficulty", 
                    value="Medium"
                )
                mcq_count = gr.Slider(
                    0, 20, value=5, step=1, 
                    label="📊 Number of MCQs"
                )
            
            gr.Markdown("### ✍️ Section B: Short Answer Questions")
            with gr.Group():
                short_difficulty = gr.Radio(
                    ["Easy", "Medium", "Hard"], 
                    label="🎚️ Short Answer Difficulty", 
                    value="Medium"
                )
                short_count = gr.Slider(
                    0, 15, value=3, step=1, 
                    label="📊 Number of Short Answer Questions"
                )
            
            gr.Markdown("### 📖 Section C: Long Answer Questions")
            with gr.Group():
                long_difficulty = gr.Radio(
                    ["Easy", "Medium", "Hard"], 
                    label="🎚️ Long Answer Difficulty", 
                    value="Medium"
                )
                long_count = gr.Slider(
                    0, 10, value=2, step=1, 
                    label="📊 Number of Long Answer Questions"
                )
            
            btn = gr.Button("✨ Generate Question Paper(s)", variant="primary", size="lg")
            
            gr.Markdown("""
            ### 📝 Instructions:
            1. Upload 1-5 PDF files containing study material
            2. Choose number of sets to generate (1-3)
            3. Configure each section:
               - Set difficulty level
               - Set number of questions
            4. Click Generate!
            
            **Note:** Set any section to 0 questions to exclude it.
            """)
        
        with gr.Column(scale=2):
            output = gr.Markdown(
                label="Generated Question Paper(s)",
                value="👋 Upload PDF files and configure settings to generate question papers..."
            )

    btn.click(
        fn=generate_question_paper,
        inputs=[
            pdf_input, 
            mcq_difficulty, mcq_count,
            short_difficulty, short_count,
            long_difficulty, long_count,
            num_sets
        ],
        outputs=output,
        show_progress="full"
    )
    
    gr.Markdown("""
    ---
    **Features:**
    - ✅ Multiple PDF support (up to 5 files)
    - ✅ Separate difficulty control for each question type
    - ✅ Customizable question count per section
    - ✅ Generate 1-3 unique question paper sets
    - ✅ Automatic answer key generation for MCQs
    - ✅ Queue system for concurrent users
    
    **Performance Tips:**
    - For faster results: Use 1-2 PDFs, fewer questions, single set
    - If timeout occurs: Reduce number of questions or sets
    - Queue position will be shown when multiple users are active
    """)

if __name__ == "__main__":
    demo.queue(
        max_size=20,  # Maximum queue size
        default_concurrency_limit=2  # Allow 2 concurrent users
    )
    demo.launch(
        show_error=True,
        share=False
    )