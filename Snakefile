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
        hebrew = "parashot/sefira/hebrew_{parasha}.json",
        english ="parashot/sefira/english_{parasha}.json"
        
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

        import arabic_reshaper
        from bidi.algorithm import get_display
        import json
        from xml.sax.saxutils import escape
        from bs4 import BeautifulSoup



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
        from reportlab.lib.enums import TA_LEFT, TA_RIGHT
        import re
        def clean_brakets(s):
                    # Use re.sub() to remove all text between {} including the braces
            result_string = re.sub(r'\{.*?\}', '', s)

            # Strip any extra spaces left over
            return re.sub(r'\s+', ' ', result_string).strip()

        # Define styles for English and Hebrew cells
        cell_eng = ParagraphStyle(
            'EnglishCellStyle',
            parent=styles['Normal'],
            alignment=TA_LEFT,    # Align the English text to the left
            valign='TOP',         # Align to the top of the cell
            fontName='NotoSansHebrew'
        )

        cell_heb = ParagraphStyle(
            'HebrewCellStyle',
            parent=styles['Normal'],
            alignment=TA_RIGHT,   # Align the Hebrew text to the right
            valign='TOP',         # Align to the top of the cell
            fontName='NotoSansHebrew'
        )

        # Prepare PDF content
        elements = []

        # Add title page
        elements.append(Paragraph(f"{row.n + 1}", title_style))
        elements.append(Paragraph(f"{bidi_hebrew} {wildcards.parasha}", subtitle_style))
        elements.append(Paragraph(f"{row.ref}", reference_style))

        # Add a page break
        elements.append(PageBreak())

        hebrew  = json.load(open(input.hebrew))['versions'][0]['text']
        english = json.load(open(input.english))['versions'][0]['text']
        verse,line = (row.ref.split()[-1].split("-")[0].split(":"))
        verse,line = int(verse),int(line)
        if type(hebrew[0])==str:
            hebrew=[hebrew]
            english=[english]
        for v_hebew,v_english in zip(hebrew,english):
            table = []
            elements.append(Paragraph(f"{verse}", title_style))

            for l_hebrew,l_english in zip(v_hebew,v_english):
                verse
                line
                l_english = Paragraph(BeautifulSoup(l_english, "lxml").text,cell_eng)
                l_hebrew  = Paragraph(BeautifulSoup(clean_brakets(l_hebrew),  "lxml").text[::-1],cell_heb)
                table.append((
                    l_hebrew,
                    line,
                    l_english))
                line+=1
            W=letter[0]
            wings = .4
            mid   = .025
            table = Table(table,colWidths=[W*wings,W*mid,W*wings])

            # Apply styles to the table
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'NotoSansHebrew'),     # Set Hebrew font for all cells
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),       # Background color for data rows
                ('GRID', (0, 0), (-1, -1), 1, colors.white),          # Add a grid to the table
                ('HALIGN', (0, 0), (-1, -1), 'MIDDLE'),                # Vertically align to middle
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),                  # Align the Hebrew column (first column) to the right
                ('ALIGN', (2, 0), (2, -1), 'LEFT'),                   # Align the English column (third column) to the left
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),                 # Center-align the middle column
                ('WORDWRAP', (0, 0), (-1, -1), 'CJK'),                # Enable word wrapping for all cells
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),                  # Vertically align all cells to the top
            ]))
            elements.append(table)
            line=1
            verse+=1       
        # Build the PDF
        elements.append(PageBreak())

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

