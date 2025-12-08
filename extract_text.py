
import sys

def search_pdf(filename, search_term):
    print(f"Searching {filename} for '{search_term}'...")
    try:
        import PyPDF2
        print("Using PyPDF2")
        reader = PyPDF2.PdfReader(filename)
        with open("pdf_content.txt", "w", encoding="utf-8") as f:
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                f.write(f"\n--- PAGE {i+1} ---\n")
                f.write(text)
        print("Dumped content to pdf_content.txt")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    search_pdf("Group_Project_Spring_2026.pdf", "Task 3")
