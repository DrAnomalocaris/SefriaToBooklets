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


rule get_parasha:
    input:
        "parashot.csv"
    output:
        "parashot/sefira/{lang}_{parasha}.json"
    run:
        import urllib.parse
        import requests
        import pandas as pd
        parashotFile = checkpoints.get_parashot_csv.get().output[0] 
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

rule make_pdf:
    input:
        table="parashot.csv",
        hebrew="parashot/sefira/hebrew_{parasha}.json",
        english="parashot/sefira/english_{parasha}.json"
    output:
        "parashot/{parasha}.pdf"
    run:
        import pandas as pd
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
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

        def remove_footnotes(s):
            # Parse the string with BeautifulSoup
            soup = BeautifulSoup(s, "html.parser")
            # Find and remove all <i> tags with class "footnote"
            for footnote_tag in soup.find_all("i", class_="footnote"):
                footnote_tag.decompose()
            # Get the cleaned up HTML
            return str(soup)
        
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
                p_english = [Paragraph(i, cell_eng) for i in split_string_without_splitting_words(BeautifulSoup(remove_footnotes(l_english), "lxml").text,50)]
                p_hebrew  = [Paragraph(i[::-1], cell_heb) for i in split_string_without_splitting_words(clean_brackets(BeautifulSoup(remove_footnotes(l_hebrew),  "lxml").text),100)]
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


rule make_book:
    input:
        pdf="parashot/{parasha}.pdf"
    output:
        "booklets/{parasha}.pdf"
    shell:
        'pdfbook2 "{input.pdf}" --paper=letter --no-crop && mv "parashot/{wildcards.parasha}-book.pdf" "{output}"'

rule all:
    input:
        lambda wildcards: expand("booklets/{parasha}.pdf", parasha=parashot_list(wildcards))

