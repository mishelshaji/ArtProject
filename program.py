import fitz  # PyMuPDF
import os
from datetime import datetime


def get_font_info(page, text_instance):
    """Extract font details from a text instance."""
    text_dict = page.get_text("dict")
    for block in text_dict["blocks"]:
        if block["type"] == 0:  # Text block
            for line in block["lines"]:
                for span in line["spans"]:
                    span_rect = fitz.Rect(span["bbox"])
                    if span_rect.intersects(text_instance):
                        # Convert color from integer to RGB tuple
                        color_int = span["color"]
                        r = (color_int >> 16) & 255
                        g = (color_int >> 8) & 255
                        b = color_int & 255
                        color_rgb = (r / 255, g / 255, b / 255)
                        return {
                            "fontname": span["font"],
                            "fontsize": span["size"],
                            "color": color_rgb
                        }
    # Default fallback
    return {"fontname": "helv", "fontsize": 11, "color": (0, 0, 0)}

def replace_multiple_in_pdf(input_pdf, output_pdf, replacements, font_file=None):
    try:
        # Open the input PDF
        pdf_document = fitz.open(input_pdf)
        
        # Load external font if provided
        if font_file and os.path.exists(font_file):
            pdf_document.insert_font(fontname="customfont", fontfile=font_file)
            fallback_font = "customfont"
        else:
            fallback_font = "helv"  # Default Helvetica
        
        # Iterate through each page
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Process each replacement
            for old_text, new_text in replacements.items():
                text_instances = page.search_for(old_text)
                
                if text_instances:
                    print(f"Page {page_num + 1}: Found '{old_text}' at {len(text_instances)} locations")
                    
                    for inst in text_instances:
                        # Get font information
                        font_info = get_font_info(page, inst)
                        
                        # Use original font if available, otherwise fallback
                        try_font = font_info["fontname"]
                        try:
                            # Test if the font is usable
                            page.insert_text((0, 0), "test", fontname=try_font, fontsize=1)
                            page.delete_text((0, 0, 10, 10))  # Clean up test
                        except Exception:
                            try_font = fallback_font
                            print(f"  Font '{font_info['fontname']}' not available, using '{try_font}'")
                        
                        # Erase original text
                        page.draw_rect(inst, color=(1, 1, 1), fill=(1, 1, 1))
                        
                        # Insert new text
                        page.insert_text(
                            (inst.x0, inst.y0),
                            new_text,
                            fontname=try_font,
                            fontsize=font_info["fontsize"],
                            color=font_info["color"]
                        )
                        print(f"  Replaced '{old_text}' with '{new_text}' using {font_info}")
        
        # Save the modified PDF
        pdf_document.save(output_pdf)
        pdf_document.close()
        print(f"Modified PDF saved as: {output_pdf}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    # File paths
    input_pdf = "certificate.pdf"
    output_pdf = "output.pdf"

    # Get the current date
    current_date = datetime.now()

    # Format the date in dd/MM/yyyy format
    formatted_date = current_date.strftime(f"\n%d/%m/%Y")
    
    # Dictionary of multiple replacements
    replacements = {
        "STUDENT": "MISHEL SHAJI",
        "INSTRUCTORNAME": "MISHEL S",
        "DATE": formatted_date
    }
    
    # Verify input file exists
    if not os.path.exists(input_pdf):
        print(f"Error: Input file '{input_pdf}' not found")
        return
    
    replace_multiple_in_pdf(input_pdf, output_pdf, replacements)

if __name__ == "__main__":
    main()