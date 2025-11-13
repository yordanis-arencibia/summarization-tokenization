import os
import time
import google.generativeai as genai
from docx import Document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
# Try both common environment variable names for the API key
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("ERROR: No API key found!")
    print("Please set one of these environment variables:")
    print("  export GOOGLE_API_KEY='your-api-key-here'")
    print("  export GEMINI_API_KEY='your-api-key-here'")
    print("\nOr get your API key from: https://makersuite.google.com/app/apikey")
    exit(1)

genai.configure(api_key=API_KEY)

# Using latest Gemini Flash for speed and cost-efficiency
# Alternative models: "gemini-2.5-pro", "gemini-2.0-flash", "gemini-pro-latest"
MODEL_NAME = "gemini-2.5-flash"

def upload_pdf_to_gemini(file_path):
    """
    Uploads a PDF file to Gemini File API for processing.
    Returns the file object after processing is complete.
    """
    print(f"--> Uploading PDF: {file_path}...")
    try:
        # Upload the file to Gemini's temporary storage
        uploaded_file = genai.upload_file(path=file_path)
        
        # Wait for the file to be ready (essential for large PDFs)
        while uploaded_file.state.name == "PROCESSING":
            print(f"    Processing {uploaded_file.name}...", end='\r')
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
            
        if uploaded_file.state.name == "FAILED":
            raise ValueError(f"File processing failed: {uploaded_file.state.name}")
            
        print(f"    Ready: {uploaded_file.display_name}")
        return uploaded_file
    except Exception as e:
        print(f"Error uploading PDF: {e}")
        return None

def extract_text_from_docx(file_path):
    """
    Extracts raw text from a .docx file locally using python-docx.
    Gemini File API is better for PDFs; text is better for DOCX.
    """
    print(f"--> Reading DOCX: {file_path}...")
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return None

def generate_summary(files_content_list):
    """
    Sends the accumulated content (file objects and text strings) to the model.
    """
    print("\nGenerating summary using Gemini...")
    
    model = genai.GenerativeModel(MODEL_NAME)
    
    # System prompt to enforce the summary structure and length constraints
    # Note: We request < 16k tokens implicitly by asking for a concise summary, 
    # though the model's output limit is typically around 8192 tokens by default.
    prompt = (
        "You are an expert analyst. Please analyze the attached documents and provided text. "
        "Create a comprehensive consolidated summary of all these files. "
        "The summary must be detailed but concise enough to fit comfortably within an output limit. "
        "Focus on: 1. Key Objectives, 2. Financial/Technical Details, 3. Risks and Conclusions."
    )
    
    # Prepare the content for the generate_content method
    # It can accept a list containing strings (prompts/text) and file objects
    request_content = [prompt]
    request_content.extend(files_content_list)
    
    # Set generation config to ensure we don't hit unexpected short limits
    # max_output_tokens limit for 1.5 Flash is typically 8192. 
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=8192, # Max standard output. This ensures it is < 16384.
        temperature=0.2,        # Low temperature for more factual summaries
    )

    response = model.generate_content(
        request_content,
        generation_config=generation_config
    )
    
    return response.text

def main():
    # --- INPUT: List your files here ---
    files_to_process = [
        "sample.pdf",       # Example PDF file
        # "notas_reunion.docx",       # Example DOCX file
        # "otro_archivo.pdf"
    ]
    
    model_inputs = []

    # 1. Process files based on extension
    for file_path in files_to_process:
        if not os.path.exists(file_path):
            print(f"Skipping: {file_path} (File not found)")
            continue
            
        if file_path.lower().endswith('.pdf'):
            pdf_obj = upload_pdf_to_gemini(file_path)
            if pdf_obj:
                model_inputs.append(pdf_obj)
                
        elif file_path.lower().endswith('.docx'):
            docx_text = extract_text_from_docx(file_path)
            if docx_text:
                # Tagging the text so the model knows which file it comes from
                labeled_text = f"\n--- Content from file: {file_path} ---\n{docx_text}"
                model_inputs.append(labeled_text)
        else:
            print(f"Unsupported format for this script: {file_path}")

    if not model_inputs:
        print("No valid files to process.")
        return

    # 2. Generate Summary
    try:
        start = time.perf_counter()
        summary = generate_summary(model_inputs)
        end = time.perf_counter()
        
        print("\n" + "="*40)
        print("SUMMARY GENERATED")
        print("="*40 + "\n")
        # print(summary)
        
        print(f"\nSummary generated in {end - start:.2f} seconds.")
        
        # Optional: Save to file
        with open("summary_output.txt", "w", encoding="utf-8") as f:
            f.write(summary)
            print("\nSummary saved to summary_output.txt")
            
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()