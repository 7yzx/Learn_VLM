from pdf2image import convert_from_path

pdf_path = '/archive/share/cql/LLM-FoR-ALL/mini_vlm/data/2003.06713v1.pdf'
images = convert_from_path(pdf_path, dpi=300) 
for i, img in enumerate(images):
    img.save(f'output_page_{i+1}.png', 'PNG')