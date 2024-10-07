from bs4 import BeautifulSoup
from openai import OpenAI
import os
from norerun import norerun
import tiktoken

checkpoint get_parashot_csv:
    output:
        "parashot.csv"
    run:
        import pandas as pd

        # Raw URL to access the CSV file
        url = 'https://raw.githubusercontent.com/Sefaria/Sefaria-Project/master/data/tmp/parsha.csv'
        # Load the CSV file into a DataFrame
        parashot = pd.read_csv(url)
        parashot.index.name="n"

        # Save the DataFrame to CSV
        parashot.to_csv(output[0], index=True)


def parashot_list(wildcards):
    import pandas as pd
    parashotFile = checkpoints.get_parashot_csv.get().output[0] 
    parashot = pd.read_csv(parashotFile)
    return parashot['en'].tolist()

def parasha_verse(wildcards):
    import pandas as pd
    parashotFile = checkpoints.get_parashot_csv.get().output[0] 
    parashot = pd.read_csv(parashotFile)
    parashot.index=parashot['en']
    return (parashot.loc[wildcards.parasha]['ref'])


def remove_footnotes_en(s):
    # Parse the string with BeautifulSoup
    if s==[]:s=""
    if type(s) == list: s = " ".join(s)
    soup = BeautifulSoup(s, "html.parser")
    
    # Iterate through all <i> tags with class "footnote"
    for footnote_tag in soup.find_all("i", class_="footnote"):
        # Get the text inside the footnote tag, strip any extra whitespace
        footnote_text = footnote_tag.get_text(strip=False)
        
        # Replace the footnote tag with its text wrapped in brackets
        footnote_tag.replace_with(f" ({footnote_text})")
    for footnote_marker in soup.find_all("sup",class_="footnote-marker"):
        footnote_marker.decompose()
    # Get the cleaned-up text without any HTML tags
    return soup.get_text()
def remove_footnotes_heb(s):
    # Parse the string with BeautifulSoup
    soup = BeautifulSoup(s, "html.parser")
    # Find and remove all <i> tags with class "footnote"
    for footnote_tag in soup.find_all("i", class_="footnote"):
        footnote_tag.decompose()
    # Get the cleaned up HTML
    return str(soup)
def trim_to_max_tokens(text, max_tokens=10000, model="gpt-3.5-turbo"):
    # Load the tokenizer for the given model
    encoding = tiktoken.encoding_for_model(model)
    
    # Tokenize the text
    tokens = encoding.encode(text)
    
    # Trim the tokens if they exceed the max_tokens limit
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    
    # Decode the tokens back to text
    trimmed_text = encoding.decode(tokens)
    return trimmed_text
@norerun
def summarize_text(text,refereces=False):
    # Get the API key from the environment variable
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    if refereces:
        text = trim_to_max_tokens( f"Please summarize the following text in no more than one paragraph, keep references in brackets indicating from which commentary it came (is at the beginning of each paragraph), Do it succintly, do not include introductions, just bulletpoint statements of specifics:\n\n{text}")
    else:
        text = trim_to_max_tokens( f"Please summarize the following text in no more than one paragraph, Do it succintly, do not include introductions, just bulletpoint statements of specifics, not as a debate between sources, but mention specifically what they say:\n\n{text}")

    chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content":text,
                }
            ],
            model="gpt-3.5-turbo",
        )
    
    return chat_completion.choices[0].message.content

rule get_parasha:
    input:
        "parashot.csv"
    output:
        "sefaria/{lang}_{parasha}.json"
    run:
        import urllib.parse
        import requests
        import pandas as pd
        parashot = pd.read_csv(parashotFile)
        parashot.index=parashot['en']
        verses = (parashot.loc[wildcards.parasha]['ref'])

        # Original verse reference
        encoded_reference = urllib.parse.quote(verses)
        url = f"https://www.sefaria.org/api/v3/texts/{encoded_reference}?version={wildcards.lang}"
        headers = {"accept": "application/json"}

        response = requests.get(url, headers=headers)
        with open(output[0], "w") as f:
            f.write(response.text)
rule get_commentary:
    input:
        parashotFile = "parashot.csv"
    output:
        "sefaria/commentary_{parasha}.json"
    run:
        import urllib.parse
        import requests
        import pandas as pd
        parashot = pd.read_csv(input.parashotFile)
        parashot.index=parashot['en']
        verses = (parashot.loc[wildcards.parasha]['ref'])
        print()
        encoded_reference = urllib.parse.quote(verses)
        url = f"https://www.sefaria.org/api/links/{encoded_reference}"
        print(url)
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        with open(output[0], "w") as f:
            f.write(response.text)

ruleorder: get_commentary>get_parasha

rule parse_commentary:
    input:
        commentary="sefaria/commentary_{parasha}.json"
    output:
        "sefaria/commentary_{parasha}.csv",
    run:
        import json
        from pprint import pprint
        import pandas as pd
        with open(input.commentary) as f:
            commentsaries = json.load(f)
        out = []
        for commentary in commentsaries:
            ref = commentary["anchorRefExpanded"][-1]
            if "Rav Hirsch on Torah" in commentary["ref"]: continue

            out.append({
                "verse" : int(ref.split()[-1].split(":")[0]),
                "line" : int(ref.split()[-1].split(":")[1]),
                "category" : commentary["category"],
                "source" : commentary["ref"],
                "text" : BeautifulSoup(remove_footnotes_en(commentary["text"]), "lxml").text
            })

        df = pd.DataFrame(out)
        df = df[df.category != "Reference"]
        df = df[df.text != ""]
        df.to_csv(output[0], index=False)

        for category in df.category.unique():
            print(category, len(df[df.category == category]))

rule make_pdf:
    input:
        table="parashot.csv",
        hebrew="sefaria/hebrew_{parasha}.json",
        english="sefaria/english_{parasha}.json"
    output:
        "parashot/{parasha}.pdf"
    run:
        import pandas as pd
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Frame
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase import pdfmetrics
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT, TA_RIGHT
        import arabic_reshaper
        from bidi.algorithm import get_display
        import json
        from xml.sax.saxutils import escape
        from bs4 import BeautifulSoup
        import re

        # Load the row for the given parasha
        row = pd.read_csv(input.table, index_col=1).loc[wildcards.parasha]

        # Register Hebrew font (path to .ttf file)
        pdfmetrics.registerFont(TTFont('NotoSansHebrew', 'src/fonts/Noto_Serif_Hebrew/NotoSerifHebrew-VariableFont_wdth,wght.ttf'))

        # Reshape and reorder the Hebrew text to display correctly
        reshaped_hebrew = arabic_reshaper.reshape(row.he)
        bidi_hebrew = get_display(reshaped_hebrew)

        # Create a PDF document
        pdf = SimpleDocTemplate(output[0], pagesize=letter)

        # Define styles for text using the registered Hebrew font
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Title'],
            alignment=1,  # Center the text
            spaceAfter=20,
            fontSize=24,
            fontName='NotoSansHebrew'
        )
        subtitle_style = ParagraphStyle(
            'SubtitleStyle',
            parent=styles['Title'],
            alignment=1,  # Center the text
            spaceAfter=20,
            fontSize=18,
            fontName='NotoSansHebrew'
        )
        reference_style = ParagraphStyle(
            'ReferenceStyle',
            parent=styles['Normal'],
            alignment=1,  # Center the text
            spaceAfter=20,
            fontSize=16,
            fontName='NotoSansHebrew'
        )

        # Utility functions for cleaning text
        def clean_brackets(s):
            # Remove all text between {} including the braces
            result_string = re.sub(r'\{.*?\}', '', s)
            # Strip any extra spaces left over
            return re.sub(r'\s+', ' ', result_string).strip()

        
        
        def split_string_without_splitting_words(s, n):
            words = s.split()  # Split the string into words
            chunks = []
            current_chunk = []

            current_length = 0
            for word in words:
                # Check if adding the next word exceeds the chunk size
                if current_length + len(word) + len(current_chunk) > n:
                    # Append the current chunk to the list and start a new one
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)
                    current_length += len(word)

            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(' '.join(current_chunk))

            return chunks
        # Define styles for English and Hebrew cells
        cell_eng = ParagraphStyle(
            'EnglishCellStyle',
            parent=styles['Normal'],
            alignment=TA_LEFT,    # Align the English text to the left
            fontName='NotoSansHebrew'
        )

        cell_heb = ParagraphStyle(
            'HebrewCellStyle',
            parent=styles['Normal'],
            alignment=TA_RIGHT,   # Align the Hebrew text to the right
            fontName='NotoSansHebrew',
            spaceAfter=0,
            leading=14  # Increase leading to improve readability of Hebrew text
        )

        # Prepare PDF content
        elements = []

        # Add title page
        elements.append(Paragraph(f"{row.n + 1}", title_style))
        elements.append(Paragraph(f"{bidi_hebrew} {wildcards.parasha}", subtitle_style))
        elements.append(Paragraph(f"{row.ref}", reference_style))

        # Add a page break
        elements.append(PageBreak())

        # Load Hebrew and English texts
        hebrew = json.load(open(input.hebrew))['versions'][0]['text']
        english = json.load(open(input.english))['versions'][0]['text']

        # Ensure texts are lists of lists
        if isinstance(hebrew[0], str):
            hebrew = [hebrew]
            english = [english]

        # Iterate through verses and add them to the table
        verse, line = map(int, row.ref.split()[-1].split("-")[0].split(":"))
        for v_hebrew, v_english in zip(hebrew, english):
            table_data = []

            elements.append(Paragraph(f"{verse}", title_style))

            for l_hebrew, l_english in zip(v_hebrew, v_english):
                # Clean and prepare Hebrew and English texts
                p_english = [Paragraph(i, cell_eng) for i in split_string_without_splitting_words(BeautifulSoup(remove_footnotes_en(l_english), "lxml").text,50)]
                p_hebrew  = [Paragraph(i[::-1], cell_heb) for i in split_string_without_splitting_words(clean_brackets(BeautifulSoup(remove_footnotes_heb(l_hebrew),  "lxml").text),100)]
                # Append Hebrew, line number, and English to the table
                table_data.append((p_hebrew, line, p_english))
                line += 1

            # Set up table layout
            W = letter[0]
            wings = 100
            mid = 0.025
            table = Table(table_data, colWidths=[W * wings, W * mid, W * wings])

            # Apply styles to the table
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'NotoSansHebrew'),  # Set Hebrew font for all cells
                ('BACKGROUND', (0, 0), (-1, -1), colors.white),    # Background color for all cells
                ('GRID', (0, 0), (-1, -1), 1, colors.white),        # Add a grid to the table
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),               # Vertically align all cells to the top
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),               # Align the Hebrew column (first column) to the right
                ('ALIGN', (2, 0), (2, -1), 'LEFT'),                # Align the English column (third column) to the left
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),              # Center-align the middle column
            ]))
            elements.append(table)
            line = 1
            verse += 1

        # Add a blank page at the end
        elements.append(PageBreak())

        # Build the PDF
        pdf.build(elements)


rule make_pdf_with_commentary:
    input:
        table="parashot.csv",
        hebrew="sefaria/hebrew_{parasha}.json",
        english="sefaria/english_{parasha}.json",
        commentary="sefaria/commentary_{parasha}.csv",
    output:
        book="parashot_commentary/{parasha}.pdf",
        expanded="parashot_commentary/{parasha}_expanded.pdf"
    run:
        import pandas as pd
        from tqdm import tqdm
        import qrcode
        import urllib.parse
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase import pdfmetrics
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT, TA_RIGHT
        from reportlab.lib.units import inch
        import arabic_reshaper
        from bidi.algorithm import get_display
        import json
        from xml.sax.saxutils import escape
        from bs4 import BeautifulSoup
        import re
        import os
        from io import BytesIO
        comments = pd.read_csv(input.commentary)
        # Load the row for the given parasha
        row = pd.read_csv(input.table, index_col=1).loc[wildcards.parasha]

        # Register Hebrew font (path to .ttf file)
        pdfmetrics.registerFont(TTFont('NotoSansHebrew', 'src/fonts/Noto_Serif_Hebrew/NotoSerifHebrew-VariableFont_wdth,wght.ttf'))

        # Reshape and reorder the Hebrew text to display correctly
        reshaped_hebrew = arabic_reshaper.reshape(row.he)
        bidi_hebrew = get_display(reshaped_hebrew)

        # Create a PDF document
        pdf    = SimpleDocTemplate(output.book,     pagesize=letter)
        pdfexp = SimpleDocTemplate(output.expanded, pagesize=letter)

        
        # Define styles for text using the registered Hebrew font
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Title'],
            alignment=1,  # Center the text
            spaceAfter=20,
            fontSize=24,
            fontName='NotoSansHebrew'
        )
        subtitle_style = ParagraphStyle(
            'SubtitleStyle',
            parent=styles['Title'],
            alignment=1,  # Center the text
            spaceAfter=20,
            fontSize=18,
            fontName='NotoSansHebrew'
        )
        reference_style = ParagraphStyle(
            'ReferenceStyle',
            parent=styles['Normal'],
            alignment=1,  # Center the text
            spaceAfter=20,
            fontSize=16,
            fontName='NotoSansHebrew'
        )
        commentary_style = ParagraphStyle(
            'ReferenceStyle',
            parent=styles['Normal'],
            alignment=TA_LEFT,  # Center the text
            spaceAfter=2,
            fontSize=8,
            fontName='NotoSansHebrew'
        )
        commentary_title_style = ParagraphStyle(
            'ReferenceStyle',
            parent=styles['Normal'],
            alignment=1,  # Center the text
            spaceAfter=2,
            fontSize=12,
            fontName='NotoSansHebrew'
        )

        # Prepare PDF content
        elements = []
        elements_expanded = []

        # Add title page
        elements.append(Paragraph(f"{row.n + 1}", title_style))
        elements.append(Paragraph(f"{bidi_hebrew} {wildcards.parasha}", subtitle_style))
        elements.append(Paragraph(f"{row.ref}", reference_style))
        
        elements_expanded.append(Paragraph(f"{row.n + 1}", title_style))
        elements_expanded.append(Paragraph(f"{bidi_hebrew} {wildcards.parasha}", subtitle_style))
        elements_expanded.append(Paragraph(f"Expanded Commentary", subtitle_style))
        elements_expanded.append(Paragraph(f"{row.ref}", reference_style))
        BOOK = row.ref.split()[0]
        # Add a page break
        elements.append(PageBreak())
        elements_expanded.append(PageBreak())
        elements.append(PageBreak())
        elements_expanded.append(PageBreak())


        #QR code bit:
        qr_url = f"https://github.com/DrAnomalocaris/SefriaToBooklets/blob/main/{urllib.parse.quote(output.expanded)}"
        # Generate the QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_url)
        qr.make(fit=True)

        # Save QR code to a BytesIO buffer
        qr_image = BytesIO()
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_img.save(qr_image)
        qr_image.seek(0)

        qr_image_element = Image(qr_image, width=2*inch, height=2*inch)
        elements.append(qr_image_element)
        elements.append(Paragraph("Expanded commentary and sources",subtitle_style))
        qr2 = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr2.add_data("https://github.com/DrAnomalocaris/SefriaToBooklets")
        qr2.make(fit=True)

        # Save QR code to a BytesIO buffer
        qr2_image = BytesIO()
        qr2_img = qr2.make_image(fill_color="black", back_color="white")
        qr2_img.save(qr2_image)
        qr2_image.seek(0)
        qr2_image_element = Image(qr2_image, width=2*inch, height=2*inch)

        elements.append(qr2_image_element)
        elements.append(Paragraph("GitHub",subtitle_style))
        elements.append(PageBreak())


        

        # Utility functions for cleaning text
        def clean_brackets(s):
            # Remove all text between {} including the braces
            result_string = re.sub(r'\{.*?\}', '', s)
            result_string = re.sub(r'\[.*?\]', '', s)
            # Strip any extra spaces left over
            return re.sub(r'\s+', ' ', result_string).strip()

        def invert_brackets(s):
            return ''.join([")" if i == "(" else "(" if i == ")" else i for i in s])

        
        def split_string_without_splitting_words(s, n):
            words = s.split()  # Split the string into words
            chunks = []
            current_chunk = []

            current_length = 0
            for word in words:
                # Check if adding the next word exceeds the chunk size
                if current_length + len(word) + len(current_chunk) > n:
                    # Append the current chunk to the list and start a new one
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)
                    current_length += len(word)

            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(' '.join(current_chunk))

            return chunks
        # Define styles for English and Hebrew cells
        cell_eng = ParagraphStyle(
            'EnglishCellStyle',
            parent=styles['Normal'],
            alignment=TA_LEFT,    # Align the English text to the left
            fontName='NotoSansHebrew'
        )

        cell_heb = ParagraphStyle(
            'HebrewCellStyle',
            parent=styles['Normal'],
            alignment=TA_RIGHT,   # Align the Hebrew text to the right
            fontName='NotoSansHebrew',
            spaceAfter=0,
            leading=14  # Increase leading to improve readability of Hebrew text
        )

 


        # Load Hebrew and English texts
        hebrew = json.load(open(input.hebrew))['versions'][0]['text']
        english = json.load(open(input.english))['versions'][0]['text']

        # Ensure texts are lists of lists
        if isinstance(hebrew[0], str):
            hebrew = [hebrew]
            english = [english]

        # Iterate through verses and add them to the table
        verse, line = map(int, row.ref.split()[-1].split("-")[0].split(":"))

        for v_hebrew, v_english in zip(hebrew, english):

            elements.append(Paragraph(f"{BOOK} {verse}", title_style))

            for l_hebrew, l_english in tqdm(list(zip(v_hebrew, v_english))):
                # Clean and prepare Hebrew and English texts
                p_english = [Paragraph(i, cell_eng) for i in split_string_without_splitting_words(BeautifulSoup(remove_footnotes_en(l_english), "lxml").text,50)]
                p_hebrew  = [Paragraph(invert_brackets(i[::-1]), cell_heb) for i in split_string_without_splitting_words(clean_brackets(BeautifulSoup(remove_footnotes_heb(l_hebrew),  "lxml").text),100)]
                # Append Hebrew, line number, and English to the table
                table_data = [[p_hebrew, line, p_english]]
                # Set up table layout
                W = letter[0]
                wings = 100
                mid = 0.025
                table = Table(table_data, colWidths=[W * wings, W * mid, W * wings])

                # Apply styles to the table
                table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, -1), 'NotoSansHebrew'),  # Set Hebrew font for all cells
                    ('BACKGROUND', (0, 0), (-1, -1), colors.white),    # Background color for all cells
                    ('GRID', (0, 0), (-1, -1), 1, colors.white),        # Add a grid to the table
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),               # Vertically align all cells to the top
                    ('ALIGN', (0, 0), (0, -1), 'RIGHT'),               # Align the Hebrew column (first column) to the right
                    ('ALIGN', (2, 0), (2, -1), 'LEFT'),                # Align the English column (third column) to the left
                    ('ALIGN', (1, 0), (1, -1), 'CENTER'),              # Center-align the middle column
                ]))
                elements.append(table)
                #add commentary style = commentary_style
                localComments = comments[(comments.verse == verse) & (comments.line == line)].sort_values('source')
                commentaries_summaries= ""
                elements_expanded.append(
                    Paragraph(f"{BOOK} {verse}:{line}",
                    ParagraphStyle(
                        'TitleStyle',
                        parent=styles['Title'],
                        alignment=1,  # Center the text
                        spaceAfter=20,
                        fontSize=24,
                        fontName='NotoSansHebrew'
                     ) 
                
                    ))
                elements_expanded.append(table)

                for category in localComments.category.unique():
                    elements_expanded.append(
                        Paragraph(
                            f"{category}", 
                            ParagraphStyle(
                                'SubtitleStyle',
                                parent=styles['Title'],
                                alignment=TA_LEFT,  # Center the text
                                spaceAfter=20,
                                fontSize=18,
                                fontName='NotoSansHebrew'
                            )
                            ))

                    commentary_text=""
                    for index, row in localComments[localComments.category == category].iterrows():
                        subCommentary=f"{row.category}|{row.source}\n{row.text}\n\n"
                        commentary_text+=subCommentary
                    summary_comment= summarize_text(commentary_text,refereces=True)
                    commentaries_summaries+=f"{category}\n{summary_comment}\n\n"
                    elements_expanded.append(
                        Paragraph(
                            summary_comment, 
                            ParagraphStyle(
                                'ReferenceStyle',
                                parent=styles['Normal'],
                                alignment=TA_LEFT,  # Center the text
                                spaceAfter=2,
                                fontSize=8,
                                fontName='NotoSansHebrew'
                            )))

                elements.append(Paragraph(summarize_text(commentaries_summaries,refereces=False), commentary_style))
                elements_expanded.append(PageBreak())
                line += 1
            line = 1
            verse += 1
        # Add a blank page at the end
        elements.append(PageBreak())
        

        # Build the PDF
        pdf.build(elements)
        pdfexp.build(elements_expanded)

rule make_book:
    input:
        pdf="parashot{comments}/{parasha}.pdf"
    output:
        "booklets{comments}/{parasha}.pdf"
    shell:
        'pdfbook2 "{input.pdf}" --paper=letter --no-crop && mv "parashot{wildcards.comments}/{wildcards.parasha}-book.pdf" "{output}"'

rule all:
    input:
        lambda wildcards: expand("booklets/{parasha}.pdf", parasha=parashot_list(wildcards))

