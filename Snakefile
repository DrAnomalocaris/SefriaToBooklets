from bs4 import BeautifulSoup
from openai import OpenAI
import os
from norerun import norerun
import tiktoken
MODEL = "gpt-3.5-turbo"
commentary_categories = [
    'Chasidut',
    'Commentary',
    'Guides',
    'Halakhah',
    'Jewish Thought',
    'Kabbalah',
    'Liturgy',
    'Midrash',
    'Mishnah',
    'Musar',
    'Quoting Commentary',
    'Responsa',
    'Second Temple',
    'Talmud',
    'Tanakh',
    'Targum',
    'Tosefta'
    ]

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


def parasha_lines(wildcards):
    import json
    from pprint import pprint
    import pandas as pd
    parashotFile = checkpoints.get_parasha.get(lang="english", parasha=wildcards.parasha).output[0] 
    parashot = json.loads(open(parashotFile).read())['versions'][0]['text']
    #parashot.index=parashot['en']
    table = pd.read_csv(checkpoints.get_parashot_csv.get().output[0])
    table.index=table['en']
    ref = table.loc[wildcards.parasha]['ref'].split()[-1].split('-')[0]
    book = table.loc[wildcards.parasha]['ref'].split()[0]
    verse,line = ref.split(':')
    verse,line = int(verse),int(line)
    out = []
    for Verse in parashot:
        for _ in Verse:
            out.append((book,verse, line))
            line +=1
        verse += 1
        line = 1
    return (out)
        
def remove_footnotes_en(s):
    # Parse the string with BeautifulSoup
    if type(s) == list: s = " ".join(s)
    if s==[]:s=""
    s = s.replace("<br>", " ")
    soup = BeautifulSoup(s, "html.parser")
    
    # Iterate through all <i> tags with class "footnote"
    for footnote_tag in soup.find_all("i", class_="footnote"):
        # Get the text inside the footnote tag, strip any extra whitespace
        footnote_text = footnote_tag.get_text(strip=False)
        
        # Replace the footnote tag with its text wrapped in brackets
        footnote_tag.replace_with(f" ({footnote_text}) ")
    for footnote_marker in soup.find_all("sup",class_="footnote-marker"):
        footnote_marker.decompose()

    # Get the cleaned-up text without any HTML tags
    return soup.get_text()

def remove_footnotes_heb(s):
    s = s.replace("<br>", " ")

    # Parse the string with BeautifulSoup
    soup = BeautifulSoup(s, "html.parser")
    # Find and remove all <i> tags with class "footnote"
    for footnote_tag in soup.find_all("i", class_="footnote"):
        footnote_tag.replace_with(" ")
    for footnote_marker in soup.find_all("sup",class_="footnote-marker"):
        footnote_marker.decompose()

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

checkpoint get_parasha:
    input:
        parashotFile="parashot.csv"
    output:
        "sefaria/{lang}_{parasha}.json"
    run:
        import urllib.parse
        import requests
        import pandas as pd
        parashot = pd.read_csv(input.parashotFile)
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
    output:
        #"sefaria/commentary_{parasha}.json",
        "sefaria/commentary/{book}_{verse}_{line}.json"
    run:
        import urllib.parse
        import requests
        import pandas as pd
        encoded_reference = urllib.parse.quote(f"{wildcards.book} {wildcards.verse}:{wildcards.line}")
        url = f"https://www.sefaria.org/api/links/{encoded_reference}"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        with open(output[0], "w") as f:
            f.write(response.text)

ruleorder: get_commentary>get_parasha

rule parse_commentary:
    input:
        #commentary="sefaria/commentary_{parasha}.json",
        parts = lambda wildcards: [f"sefaria/commentary/{book}_{verse}_{line}.json" for book, verse, line in parasha_lines(wildcards)],
    output:
        "sefaria/commentary_{parasha}.csv",
    run:
        import json
        from pprint import pprint
        import pandas as pd
        out = []
        for fname in input.parts:
            with open(fname) as f:
                commentsaries = json.load(f)
                for commentary in commentsaries:
                    ref = commentary["anchorRefExpanded"][-1]
                    #if "Rav Hirsch on Torah" in commentary["ref"]: continue

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

rule prepare_text_block_for_summary:
    input:
        commentary="sefaria/commentary_{parasha}.csv"
    output:
        "sefaria/summaries/{parasha}/text_block_{category}.pkl"
    run:
        import pandas as pd
        import pickle
        comments = pd.read_csv(input.commentary)
        comments = comments[comments.category == wildcards.category]
        out={}
        for _, row in comments.iterrows():
            if not (row.verse, row.line) in out:
                out[(row.verse, row.line)] = ""

            out[(row.verse, row.line)] += f"{row.source}\n{row.text}\n\n"

        with open(output[0], "wb") as f:
            pickle.dump(out, f)
rule get_summaries:
    input:
        "sefaria/summaries/{parasha}/text_block_{category}.pkl"
    output:
        "sefaria/summaries/{parasha}/summary_{category}.pkl"
    params:
        max_tokens_input = 15000,
        max_tokens_output = 750,
        temperature = 0.5,
        preprompt = "Please summarize the following text in no more than one paragraph,"
                    " keep references in brackets indicating from which commentary it came "
                    "(is at the beginning of each paragraph), Do it succintly, do not "
                    "include introductions, just statements of specifics.",
        model=MODEL,
    run:
        import pickle
        from openai import OpenAI
        from tqdm import tqdm
        from time import sleep

        with open(input[0], "rb") as f:
            summaries = pickle.load(f)
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        out = {}
        for verse in tqdm(summaries.keys(), total=len(summaries),desc=f"Summarizing {wildcards.category} for {wildcards.parasha}"):
            text = ( f"{params.preprompt} The text is from the {wildcards.category}.\n\n{summaries[verse]}")
            text = trim_to_max_tokens(text, max_tokens=params.max_tokens_input, model=params.model)

            chat_completion = client.chat.completions.create(
                messages=[
                            {
                                "role": "user",
                                "content":text,
                            }
                        ],
                model=params.model,
                max_tokens=params.max_tokens_output,
                temperature=params.temperature
                )
            
    
            summary= chat_completion.choices[0].message.content
            out[verse] = summary

            sleep(0.5)
        
        with open(output[0], "wb") as f:
            pickle.dump(out, f)

rule make_metasummary:
    input:
        expand("sefaria/summaries/{{parasha}}/summary_{category}.pkl", category=commentary_categories)
    output:
        "sefaria/summaries/{parasha}/meta_summary.pkl"
    params:
        max_tokens_input = 15000,
        max_tokens_output = 750,
        temperature = 0.5,
        preprompt = "Please summarize the following text in no more than one paragraph,"
                    " keep references in brackets indicating from which commentary it came "
                    "(is at the beginning of each paragraph), Do it succintly, do not "
                    "include introductions, just statements of specifics.",
        model=MODEL,
    run:
        import pickle
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        out = {}
        summaries = (pd.concat([pd.Series(pd.read_pickle(i),name=i.split("summary_")[-1].split(".")[0]) for i in input],axis=1))
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        for verse, row in tqdm(summaries.iterrows(), total=len(summaries),desc=f"Summarizing all commentary for {wildcards.parasha}"):
            block = ""
            for category,text in row.dropna().items():
                block += f"{category}: {text}\n\n"
            text = ( f"{params.preprompt}\n\n{block}")
            text = trim_to_max_tokens(text, max_tokens=params.max_tokens_input, model=params.model)

            chat_completion = client.chat.completions.create(
                messages=[
                            {
                                "role": "user",
                                "content":text,
                            }
                        ],
                model=params.model,
                max_tokens=params.max_tokens_output,
                temperature=params.temperature
                )
            summary= chat_completion.choices[0].message.content
            out[verse] =   summary
        
        with open(output[0], "wb") as f:
            pickle.dump(out, f)
        
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

""" rule make_doc_with_commentary:
    input:
        table="parashot.csv",
        hebrew="sefaria/hebrew_{parasha}.json",
        english="sefaria/english_{parasha}.json",
        commentary="sefaria/summaries/{parasha}/meta_summary.pkl",

    output:
        book="parashot_commentary/{parasha}.docx",
        #expanded="parashot_commentary/{parasha}_expanded.docx"
    run:
        import pandas as pd
        from tqdm import tqdm
        import json
        import re
        from docx import Document
        from docx.shared import Pt, Inches
        from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
        import arabic_reshaper
        from bidi.algorithm import get_display
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement

        # Function to create a table with Hebrew, Line, and English text
        def add_table_with_text(document, hebrew_text, line_number, english_text):
            # Add a table with 1 row and 3 columns
            table = document.add_table(rows=1, cols=3)
            table.autofit = True
            #set size of columns
            table.columns[0].width = Inches(4)  # Hebrew column
            table.columns[1].width = Inches(0.1)  # Line number column
            table.columns[2].width = Inches(4)  # English column

            tbl = table._tbl  # Get the table element from the table object
            tbl_pr = tbl.tblPr  # Get table properties
            
            # Create or find the table alignment element (<w:jc w:val="center"/>)
            jc = tbl_pr.xpath('w:jc')
            if not jc:
                jc = OxmlElement('w:jc')
                tbl_pr.append(jc)
            jc.set(qn('w:val'), 'left')
            # Set the width of each column 
            # Get the first row
            row = table.rows[0]
            
            # Hebrew text (Right aligned)
            cell_hebrew = row.cells[0]
            cell_hebrew.text = hebrew_text
            paragraph_hebrew = cell_hebrew.paragraphs[0]
            paragraph_hebrew.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT
            paragraph_hebrew.style = "Quote"
            # Line number (Center aligned)
            cell_line = row.cells[1]
            cell_line.text = str(line_number)
            paragraph_line = cell_line.paragraphs[0]
            paragraph_line.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            paragraph_line.style = "Quote"

            # English text (Left aligned)
            cell_english = row.cells[2]
            cell_english.text = english_text
            paragraph_english = cell_english.paragraphs[0]
            paragraph_english.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
            paragraph_english.style = "Quote"




        # Load data
        comments = pd.read_pickle(input.commentary)
 
        row = pd.read_csv(input.table, index_col=1).loc[wildcards.parasha]
        BOOK = row.ref.split()[0]

        # Reshape Hebrew text and get bidi format
        reshaped_hebrew = arabic_reshaper.reshape(row.he)
        bidi_hebrew = get_display(reshaped_hebrew)
        
        # Create a DOCX document
        doc = Document()
        
        # Function to add paragraph with specific style
        def add_paragraph_with_style(document, text, font_size=24, alignment=WD_PARAGRAPH_ALIGNMENT.CENTER, style="Normal"):
            paragraph = document.add_paragraph(text)
            paragraph.style = style

            run = paragraph.runs[0]
            run.font.size = Pt(font_size)
            paragraph.alignment = alignment
        
        # Add title page
        #add_paragraph_with_style(doc, , font_size=24)
        doc.add_heading(f"{wildcards.parasha}({row.n + 1})\n{bidi_hebrew} ",1)

        #add_paragraph_with_style(doc, f"{bidi_hebrew} {wildcards.parasha}", font_size=18,style="Title")
        add_paragraph_with_style(doc, f"{row.ref}", font_size=16)
        doc.add_page_break()

        
        # Utility functions for text cleaning
        def clean_brackets(s):
            result_string = re.sub(r'\{.*?\}', '', s)
            result_string = re.sub(r'\[.*?\]', '', s)
            return re.sub(r'\s+', ' ', result_string).strip()
        
        def invert_brackets(s):
            s = ''.join([")" if i == "(" else "(" if i == ")" else i for i in s])
            s = ''.join(["}" if i == "{" else "{" if i == "}" else i for i in s])
            return s
        # Load Hebrew and English texts
        hebrew = json.load(open(input.hebrew))['versions'][0]['text']
        english = json.load(open(input.english))['versions'][0]['text']
        
        if isinstance(hebrew[0], str):
            hebrew = [hebrew]
            english = [english]
        
        verse, line = map(int, row.ref.split()[-1].split("-")[0].split(":"))
        
        for v_hebrew, v_english in zip(hebrew, english):
            #add_paragraph_with_style(doc, f"{BOOK} {verse}", font_size=24)
            doc.add_heading(f"{BOOK} {verse}",2)

            for l_hebrew, l_english in (zip(v_hebrew, v_english)):
                cleaned_hebrew  = clean_brackets(BeautifulSoup(remove_footnotes_heb(l_hebrew),  "lxml").text)
                cleaned_english = BeautifulSoup(remove_footnotes_heb(l_english), "lxml").text
                add_table_with_text(doc, cleaned_hebrew, line, cleaned_english)

                #add_paragraph_with_style(doc, cleaned_hebrew, font_size=14, alignment=WD_PARAGRAPH_ALIGNMENT.RIGHT)
                #add_paragraph_with_style(doc, cleaned_english, font_size=14, alignment=WD_PARAGRAPH_ALIGNMENT.LEFT)
                doc.add_paragraph( comments[(verse, line)], style="Normal")

                
                #add_paragraph_with_style(doc, comments[(verse, line)], font_size=10, alignment=WD_PARAGRAPH_ALIGNMENT.LEFT)

                line += 1
            line = 1
            verse += 1
            doc.add_page_break()

        
        # Save DOCX files
        doc.save(output.book) """
""" 
rule make_the_BOOK:
    input:
        lambda wildcards: expand("parashot_commentary/{parasha}.docx", parasha=["Balak","V'Zot HaBerachah"]), #parashot_list(wildcards)
    output:
        book = "BOOK.docx"
    run:
        from docx import Document

        # Function to append the content of one document to another
        def append_doc(source_doc, target_doc):
            # Append all paragraphs
            for paragraph in source_doc.paragraphs:
                target_doc.add_paragraph(paragraph.text, style=paragraph.style)
            
            # Append all tables
            for table in source_doc.tables:
                new_table = target_doc.add_table(rows=0, cols=len(table.columns))
                for row in table.rows:
                    cells = new_table.add_row().cells
                    for i, cell in enumerate(row.cells):
                        cells[i].text = cell.text
            
        # Function to merge multiple docx files
        def merge_documents(files):
            # Create a new document
            merged_document = Document()
            
            # Iterate over the list of files and append their content to the merged document
            for i, file in enumerate(files):
                # Open each document
                doc = Document(file)
                
                # Add a page break between files (except before the first document)
                if i > 0:
                    merged_document.add_page_break()
                
                # Append the content of the current document
                append_doc(doc, merged_document)
            
            # Save the merged document
            merged_document.save(output.book)

        # Example usage
        merge_documents(input)
 """


rule make_MD_with_commentary:
    input:
        table="parashot.csv",
        hebrew="sefaria/hebrew_{parasha}.json",
        english="sefaria/english_{parasha}.json",
        commentary="sefaria/summaries/{parasha}/meta_summary.pkl",

    output:
        book="parashot_commentary/{parasha}.MD",
        #expanded="parashot_commentary/{parasha}_expanded.docx"
    run:
        import pandas as pd
        from tqdm import tqdm
        import json
        import re
        import arabic_reshaper
        from bidi.algorithm import get_display

        # Load data
        comments = pd.read_pickle(input.commentary)
 
        row = pd.read_csv(input.table, index_col=1).loc[wildcards.parasha]
        BOOK = row.ref.split()[0]

        # Reshape Hebrew text and get bidi format
        reshaped_hebrew = arabic_reshaper.reshape(row.he)
        bidi_hebrew = get_display(reshaped_hebrew)
        
        # Create a DOCX document
        doc = ""
        

        
        # Add title page

        doc += f"\n# **{row.n + 1}**: {bidi_hebrew[::-1]}|{wildcards.parasha}\n \n"
        doc += f"\n  {row.ref} \n"
        doc += '<div style="page-break-after: always;"></div>\n'

        
        # Utility functions for text cleaning
        def clean_brackets(s):
            result_string = re.sub(r'\{.*?\}', '', s)
            result_string = re.sub(r'\[.*?\]', '', result_string)
            result_string = re.sub(r'\(.*?\)', '', result_string)
            return re.sub(r'\s+', ' ', result_string).strip()
        
        def invert_brackets(s):
            s = ''.join([")" if i == "(" else "(" if i == ")" else i for i in s])
            s = ''.join(["}" if i == "{" else "{" if i == "}" else i for i in s])
            return s
        # Load Hebrew and English texts
        hebrew = json.load(open(input.hebrew))['versions'][0]['text']
        english = json.load(open(input.english))['versions'][0]['text']
        
        if isinstance(hebrew[0], str):
            hebrew = [hebrew]
            english = [english]
        
        verse, line = map(int, row.ref.split()[-1].split("-")[0].split(":"))
        
        for v_hebrew, v_english in zip(hebrew, english):
            #add_paragraph_with_style(doc, f"{BOOK} {verse}", font_size=24)
            doc += f"\n## {BOOK} {verse}\n"
            for l_hebrew, l_english in (zip(v_hebrew, v_english)):
                cleaned_hebrew  = (clean_brackets(BeautifulSoup(remove_footnotes_heb(l_hebrew),  "lxml").text))
                cleaned_english = BeautifulSoup(remove_footnotes_heb(l_english), "lxml").text
                doc += f"\n|{cleaned_hebrew}|{line}|{cleaned_english}|\n"
                doc += "|--:|:-:|:--|\n"
                doc += f"\n{comments[(verse, line)]} \n"
                
 
                line += 1
            line = 1
            verse += 1
            doc += '<div style="page-break-after: always;"></div>\n'
        doc += "\n\n\n"
        # Save MD files
        with open(output.book, "w") as f:
            f.write(doc)


rule make_book:
    input:
        pdf="parashot{comments}/{parasha}.pdf"
    output:
        "booklets{comments}/{parasha}.pdf"
    shell:
        'pdfbook2 "{input.pdf}" --paper=letter --no-crop && mv "parashot{wildcards.comments}/{wildcards.parasha}-book.pdf" "{output}"'

rule make_the_BOOK_MD:
    input:
        "Introduction.MD",
        lambda wildcards: expand("parashot_commentary/{parasha}.MD", parasha=parashot_list(wildcards)), #
    output:
        book = temporary(".BOOK.MD")
    run:

        out = ""
        for i in input:
            with open(i) as f:
                out += f.read()
        with open(output.book, "w") as f:
            f.write(out)

rule md_to_docx:
    input:
        md=".BOOK.MD"
    output:
        docx=temporary(".BOOK.docx")
    shell:
        'pandoc "{input.md}" -o "{output.docx}" -f markdown -t docx'

rule fix_docx:
    input:
        docx=".BOOK.docx"
    output:
        docx="BOOK.docx"
    run:

        from docx import Document
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        
        # Function to set vertical alignment for a cell
        def set_vertical_alignment(cell, alignment="top"):
            tc = cell._tc  # Access the underlying XML for the table cell
            tcPr = tc.get_or_add_tcPr()  # Get or create the table cell properties (tcPr)
            
            # Create or find the vertical alignment element
            vAlign = tcPr.find(qn('w:vAlign'))
            if vAlign is None:
                vAlign = OxmlElement('w:vAlign')
                tcPr.append(vAlign)
            
            # Set the alignment value ("top", "center", "bottom")
            vAlign.set(qn('w:val'), alignment)

        # Function to set table borders to white (or transparent)
        def set_table_borders_white(table):
            tbl = table._tbl  # Access the underlying XML for the table
            
            # Get or create the table properties (tblPr)
            tblPr = tbl.tblPr
            
            # Create or access the borders element (tblBorders)
            tblBorders = tblPr.find(qn('w:tblBorders'))
            if tblBorders is None:
                tblBorders = OxmlElement('w:tblBorders')
                tblPr.append(tblBorders)
            
            # Modify or create the borders: top, left, bottom, right, insideH (horizontal), insideV (vertical)
            for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
                border = tblBorders.find(qn(f'w:{border_name}'))
                if border is None:
                    border = OxmlElement(f'w:{border_name}')
                    tblBorders.append(border)
                
                # Set border color to white and border size to 0 to make it effectively transparent
                border.set(qn('w:val'), 'single')
                border.set(qn('w:sz'), '4')  # Border size (use 0 for completely invisible)
                border.set(qn('w:space'), '0')
                border.set(qn('w:color'), 'FFFFFF')  # Set to white
        # Open the document
        doc = Document(input.docx)

        # Iterate through all the tables in the document
        for table in doc.tables:
            # Set the border color to white/transparent
            set_table_borders_white(table)
            
            # Iterate through all rows and cells
            for row in table.rows:
                for cell in row.cells:
                    # Set each cell to be aligned to the top
                    set_vertical_alignment(cell, alignment="top")
        # Save the modified document
        doc.save(output.docx)

rule all:
    input:
        lambda wildcards: expand("booklets/{parasha}.pdf", parasha=parashot_list(wildcards))

