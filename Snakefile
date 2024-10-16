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
    parashot = parashot[parashot['ref'].str.contains(wildcards.book)]
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
    parashotFile = checkpoints.get_parasha.get(lang="english", parasha=wildcards["parasha"]).output[0] 
    parashot = json.loads(open(parashotFile).read())['versions'][0]['text']
    if type(parashot[0])==str:
        parashot = [parashot]
    #parashot.index=parashot['en']
    table = pd.read_csv(checkpoints.get_parashot_csv.get().output[0])
    table.index=table['en']
    ref = table.loc[wildcards["parasha"]]['ref'].split()[-1].split('-')[0]
    book = table.loc[wildcards["parasha"]]['ref'].split()[0]
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
        "sefaria/commentary/{book}/{verse}/{line}.json"
    run:
        import urllib.parse
        import requests
        import pandas as pd
        encoded_reference = urllib.parse.quote(f"{wildcards.book} {wildcards.verse}:{wildcards.line}")
        url = f"https://www.sefaria.org/api/links/{encoded_reference}"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        if response.text.startswith('{"error":'):
            raise Exception(response.text)
        with open(output[0], "w") as f:
            f.write(response.text)

ruleorder: get_commentary>get_parasha
rule parse_commentary:
    input:
        #commentary="sefaria/commentary_{parasha}.json",
        parts = lambda wildcards: [f"sefaria/commentary/{book}/{verse:02}/{line:03}.json" for book, verse, line in parasha_lines(wildcards)],
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
        with open(input.commentary) as f:
            comments = pd.read_csv(f)
        comments = comments[comments.category == wildcards.category]
        out={}
        for _, row in comments.iterrows():
            if not (int(row.verse), int(row.line)) in out:
                out[(int(row.verse), int(row.line))] = ""

            out[(int(row.verse), int(row.line))] += f"{row.source}\n{row.text}\n\n"

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



rule make_Elliott_dictionary:
    input:
        "ElliottFriedman.txt",
    output:
        "ElliottFriedman.pk"
    run:
        import pickle

        REF = {}
        def keepOnlyNumbers(s):
            return int("".join([i for i in s if i.isdigit() ]))
            


        with open(input[0]) as f:
            book=""
            for line in f:
                if line == "\n":
                    continue
                if line.startswith(">"):
                    book = line[1:].strip()
                    continue
                autor,parts = line.split("~")
                for part in parts.split(";"): 
                    if part.strip() == "": continue
                    try:
                        verse,a = part.split(":")
                    except:
                        print(book,repr(a))
                        raise ValueError(part)
                    verse = keepOnlyNumbers(verse)
                    for b in a.split(","):
                        if "-" in b:
                            try:
                                start,end = b.split("-")
                            except:
                                print(book,repr(b))
                                raise ValueError(b)
                            start = keepOnlyNumbers(start)
                            end = keepOnlyNumbers(end)
                            for c in range(start,end+1):
                                if not (book,verse,c) in REF:
                                    REF[(book,verse,c)]=[]
                                REF[(book,verse,c)].append(autor)
                        else:
                            b = keepOnlyNumbers(b)
                            if not (book,verse,b) in REF:
                                REF[(book,verse,b)]=[]
                            REF[(book,verse,b)].append(autor)
        with open(output[0], "wb") as f:
            pickle.dump(REF, f)


rule make_MD_with_commentary:
    input:
        table="parashot.csv",
        hebrew="sefaria/hebrew_{parasha}.json",
        english="sefaria/english_{parasha}.json",
        commentary="sefaria/summaries/{parasha}/meta_summary.pkl",
        ElliottFriednan="ElliottFriedman.pk",

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
        ElliottFriednan = pd.read_pickle(input.ElliottFriednan)
 
        row = pd.read_csv(input.table, index_col=1).loc[wildcards.parasha]
        BOOK = row.ref.split()[0]

        # Reshape Hebrew text and get bidi format
        reshaped_hebrew = arabic_reshaper.reshape(row.he)
        bidi_hebrew = get_display(reshaped_hebrew)
        
        # Create a DOCX document
        doc = ""
        

        
        # Add title page

        doc += f"\n# **{row.n + 1}**: {bidi_hebrew[::-1]}|{wildcards.parasha} ({row.ref})\n \n"

        
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
                doc += f"\n|{cleaned_hebrew}|{line} {' '.join(ElliottFriednan.get((BOOK,verse,line),['?']))}|{cleaned_english}|\n"
                doc += "|--:|:-:|:--|\n"
                if (verse, line) in comments.keys():
                    doc += f"\n{comments[(verse, line)]} \n"
                
 
                line += 1
            line = 1
            verse += 1
        doc += "\n\n\n"
        # Save MD files
        with open(output.book, "w") as f:
            f.write(doc)




rule make_the_BOOK_MD:
    input:
        "Introduction.MD",
        lambda wildcards: expand("parashot_commentary/{parasha}.MD", parasha=parashot_list(wildcards)), #
        
    output:
        book = temporary(".BOOK_{book}.MD")
    run:

        out = ""
        for i in input:
            with open(i) as f:
                out += f.read()
        with open(output.book, "w") as f:
            f.write(out)

rule md_to_docx:
    input:
        md=".BOOK_{book}.MD"
    output:
        docx=temporary(".BOOK_{book}.docx")
    shell:
        'pandoc "{input.md}" -o "{output.docx}" -f markdown -t docx'

rule fix_docx:
    input:
        docx=".BOOK_{book}.docx"
    output:
        docx="BOOK_{book}.docx"
    run:

        from docx import Document
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Inches

        # Open the document
        doc = Document(input.docx)
       
        # Set the page width and height
        doc.sections[0].page_width      = Inches(8.5)
        doc.sections[0].page_height     = Inches(11)
        doc.sections[0].top_margin      = Inches(0.25)
        doc.sections[0].bottom_margin   = Inches(0.25)


        # Access the "Heading 1" style
        for style in doc.styles:
            if style.name == "Heading 1":

                # Modify font settings
                font = style.font
                font.bold = True
                font.size = Pt(16)  # Optional: Set the font size
                font.color.rgb = RGBColor(0, 0, 0)  # Set the font color to black

                # Modify paragraph settings
                paragraph_format = style.paragraph_format
                paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER  # Center the text
                paragraph_format.space_after = Pt(2)  # Optional: Adjust space after paragraph

                # Add a page break before every Heading 1 paragraph
                style.paragraph_format.page_break_before = True
            elif style.name == "Heading 2":

                # Modify font settings
                font = style.font
                font.bold = True
                font.size = Pt(14)  # Optional: Set the font size
                font.color.rgb = RGBColor(100, 100, 100)  # Set the font color to black

                # Modify paragraph settings
                paragraph_format = style.paragraph_format
                paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER  # Center the text
                paragraph_format.space_after = Pt(12)  # Optional: Adjust space after paragraph

                # Add a page break before every Heading 1 paragraph
                style.paragraph_format.page_break_before = False   
            elif style.name == "Compact":
                font = style.font
                font.name = "SBL Hebrew"
                paragraph_format = style.paragraph_format
                paragraph_format.keep_together = True  # Ensure paragraph stays on the same page


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

        # Iterate through all the tables in the document
        for table in doc.tables:
            # Set the border color to white/transparent
            set_table_borders_white(table)
            
            # Iterate through all rows and cells
            for row in table.rows:
                # Access the row properties and disable splitting across pages
                row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
                for cell in row.cells:
                    # Set each cell to be aligned to the top
                    set_vertical_alignment(cell, alignment="top")

        # Save the modified document
        doc.save(output.docx)
        print("Done!")
        print("")
        print("remember to do these things in word, it is too complacated to do in snakemake:")
        print("    - Pages numbers (centered).")
        print("    - Table of contents, only one level.") 
        print("    - Save as PDF.")
        print("")
        print("Ready for the presses!")
        print("")
        print("That all folks!")

rule make_mega_json:
    output:
        json = "src/mega_{parasha}.json",

    input:
        meta_summary = "sefaria/summaries/{parasha}/meta_summary.pkl",
        summaries    = expand("sefaria/summaries/{{parasha}}/summary_{source}.pkl", source=commentary_categories),
        commentary   = "sefaria/commentary_{parasha}.csv",
        english      ="sefaria/english_{parasha}.json",
        hebrew       = "sefaria/hebrew_{parasha}.json",
    run:
        import pandas as pd
        from sortedcontainers import SortedDict
        import pickle
        import json
        from tqdm import tqdm
        import itertools



        # Create the main HTML document
        meta = parasha_lines(wildcards)
        commentary = pd.read_csv(input.commentary)
        with doc.body:

            out = SortedDict()
            with open(input.meta_summary, "rb") as f:
                meta_summary = (pickle.load(f))
            with open(input.english) as f: english = list(itertools.chain.from_iterable(json.load(f)['versions'][0]['text']))
            with open(input.hebrew) as f:  hebrew  = list(itertools.chain.from_iterable(json.load(f)['versions'][0]['text']))
            summaries = {}
            for source,_summary in zip(commentary_categories,input.summaries):
                with open(_summary, "rb") as f:
                    summaries[source] = (pickle.load(f))
            

            for (book, verse, line),hebrew_line, english_line in tqdm(zip(meta,hebrew, english), total=len(meta)):

                sub1 = SortedDict()
                for source,s in summaries.items():
                    c = (commentary[(commentary.category == source) & (commentary.verse == verse) & (commentary.line == line)])

                    c = (SortedDict(zip(c["source"], c["text"])))
                    if (verse, line) in s:  
                        sub1[source] = {
                            "summary":s[(verse, line)],
                            "commentaries": c
                            }
                if not verse in out:
                    out[verse] = SortedDict()

                out[verse][line] = {
                    "hebrew": hebrew_line, 
                    "english": english_line, 
                    "summary": meta_summary[(verse, line)],
                    "commentaries": sub1}

        with open(output.json, "w") as f: 
            json.dump(dict(out), f, indent=2)   


  

rule make_parasha_html:
    output:
        html = "parasha_{parasha}.html"

    input:
        meta_summary = "sefaria/summaries/{parasha}/meta_summary.pkl",
        summaries    = expand("sefaria/summaries/{{parasha}}/summary_{source}.pkl", source=commentary_categories),
        commentary   = "sefaria/commentary_{parasha}.csv",
        english      ="sefaria/english_{parasha}.json",
        hebrew       = "sefaria/hebrew_{parasha}.json",
        ElliottFriedman="ElliottFriedman.pk",

    run:
        import pandas as pd
        from sortedcontainers import SortedDict
        import pickle
        import json
        from tqdm import tqdm
        import itertools
        import dominate
        from dominate.tags import h1,h2,h3,h4,style,script,table,div,tr,td,tbody,p,details,summary,meta,link
        from dominate.util import raw
        ElliottFriedman = pd.read_pickle(input.ElliottFriedman)
                # Utility functions for cleaning text
        def clean_brackets(s):
            # Remove all text between {} including the braces
            result_string = re.sub(r'\{.*?\}', '', s)
            # Strip any extra spaces left over
            return re.sub(r'\s+', ' ', result_string).strip()

        # Create the main HTML document
        doc = dominate.document(title=wildcards.parasha)
        with doc.body: h1(wildcards.parasha)
        with doc.head:
            meta(charset="UTF-8")
            meta(name="viewport", content="width=device-width, initial-scale=1.0")
            
            # Link to an external CSS file
            link(rel="stylesheet", href="src/styles.css")
            
        meta = parasha_lines(wildcards)
        with open(input.commentary) as f:
            commentary = pd.read_csv(f)
        with doc.body:

            out = SortedDict()
            with open(input.meta_summary, "rb") as f:
                meta_summary = (pickle.load(f))
            with open(input.english) as f: english = list(itertools.chain.from_iterable(json.load(f)['versions'][0]['text']))
            with open(input.hebrew) as f:  hebrew  = list(itertools.chain.from_iterable(json.load(f)['versions'][0]['text']))
            summaries = {}
            for source,_summary in zip(commentary_categories,input.summaries):
                with open(_summary, "rb") as f:
                    summaries[source] = (pickle.load(f))
            

            for (book, verse, line),hebrew_line, english_line in tqdm(zip(meta,hebrew, english), total=len(meta)):
                if (line == 1) or ((book, verse, line) == meta[0]):
                    h2(f"{book}:{verse}:{line}",style="text-align: center;")
                with details():
                    summary(table(tbody(tr(
                        [
                            td(raw(clean_brackets(BeautifulSoup(remove_footnotes_heb(hebrew_line),  "lxml").text)),    style="text-align: right;   vertical-align: top;padding: 10px;"),
                            td(f'{line}\n{" ".join(ElliottFriedman.get((book, verse, line),[]))}',           style="text-align: center;  vertical-align: top;padding: 10px;"),
                            td(raw(clean_brackets(BeautifulSoup(remove_footnotes_en(english_line),  "lxml").text)),   style="text-align: left;    vertical-align: top;padding: 10px;")
                        ])),style="margin: 0 auto; width: 50%; border-collapse: collapse;"
                        )
                    )
                    with details():
                        if (verse, line) in meta_summary:
                            summary(meta_summary[(verse, line)])


                            for source,s in summaries.items():
                                c = (commentary[(commentary.category == source) & (commentary.verse == verse) & (commentary.line == line)])

                                c = (SortedDict(zip(c["source"], c["text"])))
                                if (verse, line) in s:  
                                    with details():
                                        summary(h4(source))
                                        with details():
                                            summary(s[(verse, line)])
                                            for a,b in c.items():
                                                with details():
                                                    summary(h4(a))
                                                    p(b)



        with open(output.html, 'w') as f:
            f.write(doc.render())   


rule make_index_html:
    input:
        parashot = "parashot.csv"
    output:
        html = "index.html"
    run:
        import pandas as pd
        from sortedcontainers import SortedDict
        import pickle
        import json
        from tqdm import tqdm
        import itertools
        import dominate
        from dominate.tags import h1,h2,h3,h4,style,script,table,div,tr,td,tbody,p,details,summary,meta,link,a
        from dominate.util import raw
        df = pd.read_csv(input.parashot)
        df['book'] = df['ref'].apply(lambda x: x.split()[0])
        doc = dominate.document(title="Sefaria Commentary")
        with doc.head:
            meta(charset="UTF-8")
            meta(name="viewport", content="width=device-width, initial-scale=1.0")
            
            # Link to an external CSS file
            link(rel="stylesheet", href="src/styles.css")
        with doc.body:
            for book in df['book'].unique():
                h1(book)
                for i,row in df[df['book'] == book].iterrows():
                    a(p(f'{row["n"]}: {row["en"]}'),
                        href=f"parasha_{row.en}.html")


        print(df)
        with open(output.html, 'w') as f:
            f.write(doc.render())   


rule make_book_html:
    #The html file becomes too large to display
    output:
        html = "book_{book}.html"
    input:
        meta_summary    = lambda wildcards: expand("sefaria/summaries/{parasha}/meta_summary.pkl",parasha = parashot_list(wildcards) ),
        summaries       = lambda wildcards: expand("sefaria/summaries/{parasha}/summary_{source}.pkl",parasha=parashot_list(wildcards), source=commentary_categories),
        commentary      = lambda wildcards: expand("sefaria/commentary_{parasha}.csv",parasha=parashot_list(wildcards)),
        english         = lambda wildcards: expand("sefaria/english_{parasha}.json",parasha=parashot_list(wildcards)),
        hebrew          = lambda wildcards: expand("sefaria/hebrew_{parasha}.json",parasha=parashot_list(wildcards)),
        ElliottFriednan ="ElliottFriedman.pk",
    run:
        import pandas as pd
        from sortedcontainers import SortedDict
        import pickle
        import json
        from tqdm import tqdm
        import itertools
        import dominate
        from dominate.tags import h1,h2,h3,h4,style,script,table,div,tr,td,tbody,p,details,summary,meta,link
        from dominate.util import raw
        ElliottFriednan = pd.read_pickle(input.ElliottFriednan)

                # Utility functions for cleaning text
        def clean_brackets(s):
            # Remove all text between {} including the braces
            result_string = re.sub(r'\{.*?\}', '', s)
            # Strip any extra spaces left over
            return re.sub(r'\s+', ' ', result_string).strip()

        # Create the main HTML document
        doc = dominate.document(title=wildcards.book)
        with doc.body: h1(wildcards.book)
        with doc.head:
            meta(charset="UTF-8")
            meta(name="viewport", content="width=device-width, initial-scale=1.0")
            
            # Link to an external CSS file
            link(rel="stylesheet", href="src/styles.css")
        for parasha in tqdm(parashot_list(wildcards)):
            with doc.body:
                with details():
                    summary(h2(parasha))
                    meta = parasha_lines({"parasha":parasha})
                    with open(f"sefaria/commentary_{parasha}.csv") as f:
                        commentary = pd.read_csv(f)
                    out = SortedDict()
                    with open(f"sefaria/summaries/{parasha}/meta_summary.pkl", "rb") as f:
                        meta_summary = (pickle.load(f))
                    with open(f"sefaria/english_{parasha}.json") as f: english = list(itertools.chain.from_iterable(json.load(f)['versions'][0]['text']))
                    with open(f"sefaria/hebrew_{parasha}.json") as f:  hebrew  = list(itertools.chain.from_iterable(json.load(f)['versions'][0]['text']))
                    summaries = {}
                    for source in commentary_categories:
                        with open(f"sefaria/summaries/{parasha}/summary_{source}.pkl", "rb") as f:
                            summaries[source] = (pickle.load(f))
                    
                    for (book, verse, line),hebrew_line, english_line in zip(meta,hebrew, english):
                        EF = ElliottFriednan.get((book, verse, line))
                        if (line == 1) or ((book, verse, line) == meta[0]):
                            h2(f"{book}:{verse}:{line}",style="text-align: center;")
                        with details():
                            summary(table(tbody(tr(
                                [
                                    td(raw(clean_brackets(BeautifulSoup(remove_footnotes_heb(hebrew_line),  "lxml").text)),    style="text-align: right;   vertical-align: top;padding: 10px;"),
                                    td(f'{line}\n{EF}',           style="text-align: center;  vertical-align: top;padding: 10px;"),
                                    td(raw(clean_brackets(BeautifulSoup(remove_footnotes_en(english_line),  "lxml").text)),   style="text-align: left;    vertical-align: top;padding: 10px;")
                                ]))
                                ,style="margin: 0 auto; width: 50%; border-collapse: collapse;"
                                )
                            )
                            with details():
                                if (verse, line) in meta_summary:
                                    summary(meta_summary[(verse, line)])
                                    for source,s in summaries.items():
                                        c = (commentary[(commentary.category == source) & (commentary.verse == verse) & (commentary.line == line)])
                                        c = (SortedDict(zip(c["source"], c["text"])))
                                        if (verse, line) in s:  
                                            with details():
                                                summary(h4(source))
                                                with details():
                                                    summary(s[(verse, line)])
                                                    for a,b in c.items():
                                                        with details():
                                                            summary(h4(a))
                                                            p(b)



        with open(output.html, 'w') as f:
            f.write(doc.render())   

  

rule make_web:
    input:
        "index.html",
        lambda wildcards:expand("parasha_{parasha}.html",parasha=parashot_list(wildcards))
    output:
        "done_{book}_page.txt"
    shell:
        "touch {output}"
rule all:
    input:
        expand("BOOK_{book}.docx",
            book=[
                "Genesis",
                "Exodus",
                "Leviticus",
                "Numbers",
                "Deuteronomy"
            ])

