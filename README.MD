# Parasha Booklet Generator

This repository contains a Snakemake workflow that automates the creation of bilingual booklets for weekly Torah readings (parashot). The workflow fetches the text of each parasha in both Hebrew and English from the Sefaria API, formats the text into a PDF booklet, adds GPT-generated summaries of the commentary, and generates the final output in a ready-to-print format.

## Features

- **Data Collection**: Automatically downloads the list of parashot from Sefaria and fetches the Hebrew and English texts for each portion.
- **Bilingual PDF Generation**: Generates PDFs with Hebrew and English side-by-side, using ReportLab to ensure proper text alignment and styling.
- **GPT Commentary Summaries**: Integrates GPT-generated summaries of the commentary to provide additional insights and understanding for each parasha.
- **Text Wrapping and Alignment**: Correctly handles Hebrew text direction and alignment with `bidi` and `arabic_reshaper` libraries.
- **Booklet Format**: Uses `pdfbook2` to generate a ready-to-print booklet format from the generated PDF.
- **Documentary Hypothesis**: Each line includes the source according to the documentary hypothesis.

## Requirements

- **Python 3.x**
- **Snakemake**
- **ReportLab**: For PDF generation.
- **Requests**: For API requests.
- **BeautifulSoup4**: For HTML parsing.
- **bidi** and **arabic_reshaper**: For handling Hebrew text alignment.
- **pdfbook2**: For converting PDFs into booklet format.
- **OpenAI**: For generating GPT commentary summaries.

### Python Libraries

You can install the required Python libraries using:

```sh
pip install snakemake reportlab requests beautifulsoup4 python-bidi arabic-reshaper openai
```

## Usage

1. **Clone the Repository**:

   ```sh
   git clone https://github.com/DrAnomalocaris/SefriaToBooklets.git
   cd SefriaToBooklets
   ```
2. **Set Up OpenAI API Key**:
   Set the OpenAI API key as an environment variable to allow access for generating GPT summaries.

   ```sh
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```

   Alternatively, you can add this line to your `.bashrc` or `.zshrc` file to set the key permanently.
3. **Run the Workflow for a Specific Parasha**:
   You can use Snakemake to run the workflow for a specific parasha by specifying the parasha name and ensuring the OpenAI API key is set.

   ```sh
   OPENAI_API_KEY="your_openai_api_key_here" snakemake -c1 parashot/{parasha}.pdf
   ```

   Replace `{parasha}` with the specific parasha name you want to generate. For example:

   ```sh
   OPENAI_API_KEY="your_openai_api_key_here" snakemake -c1 parashot/Bereishit.pdf
   ```
4. **Run the Workflow for All Parashot**:
   Use Snakemake to run the entire workflow and generate booklets for all parashot.

   ```sh
   snakemake -c1 all
   ```

   The `-c1` flag indicates that only one job will run at a time. Increase this number if you want parallel execution.
5. **Generated Output**:

   - The generated PDF booklets are stored in the `booklets/` directory.
     Use Snakemake to run the entire workflow and generate booklets for all parashot.

   ```sh
   snakemake -c1 all
   ```

   The `-c1` flag indicates that only one job will run at a time. Increase this number if you want parallel execution.
6. **Generated Output**:

   - The generated PDF booklets are stored in the `booklets/` directory.

## Workflow Overview

- **Checkpoint: `get_parashot_csv`**
  - Downloads the list of parashot from Sefaria.
- **Rule: `get_parasha`**
  - Fetches Hebrew and English texts for a given parasha from the Sefaria API.
- **Rule: `generate_summary`**
  - Uses GPT to generate a summary of the commentary for each parasha.
- **Rule: `make_pdf`**
  - Generates a bilingual PDF for the parasha, with proper alignment of Hebrew and English texts, and includes the GPT-generated summaries.
- **Rule: `make_book`**
  - Converts the generated PDF into a ready-to-print booklet format using `pdfbook2`.

## Example Code for Generating GPT Commentary Summaries

Here is an example of how to use the OpenAI API to generate a summary of a commentary:

```python
import os
import openai

def generate_summary(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=f"Please summarize the following commentary:

{text}",
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Example usage
commentary_text = "This is an example of a commentary text that needs summarizing."
summary = generate_summary(commentary_text)
print(summary)
```

## Key Features and Implementation Details

- **Hebrew and English Text Handling**: The workflow makes use of `bidi` and `arabic_reshaper` to ensure that Hebrew text is rendered correctly and wraps appropriately in multi-line paragraphs.
- **GPT Commentary Summaries**: Summaries of the commentary are generated using OpenAI's GPT model, providing concise explanations to enhance understanding of the parasha.
- **Page Layout**: The generated PDFs include headers, verse numbers, formatted Hebrew and English text side-by-side, and GPT summaries for easy reading.
- **Blank Page Addition**: Adds an extra blank page at the end of each PDF to facilitate printing as a booklet.

## Customization

- **Fonts**: The workflow uses `NotoSansHebrew` for Hebrew text. You can change the font by modifying the path in the `make_pdf` rule.
- **Translations**: The Sefaria API provides multiple translations. The workflow currently uses a specific translation for English text, but you can modify it by changing the `version` parameter in the URL.
- **GPT Model**: The workflow uses OpenAI's GPT-3.5 model to generate commentary summaries. You can modify the prompt or model parameters to customize the summaries.

## Potential Improvements

- **Error Handling**: Implement better error handling for network failures and missing data to make the workflow more robust.
- **Parallel Execution**: Increase the number of parallel jobs (`-j`) to speed up the processing of multiple parashot.
- **Configuration File**: Introduce a configuration file to make font paths, translation options, and other parameters easily customizable.
- **Advanced Summary Customization**: Allow users to provide custom prompts for generating GPT summaries to tailor the content to their audience.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for suggestions or bug reports.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- **Sefaria** for providing access to their API and the text data used in this project.
- **ReportLab** for providing the tools to generate well-formatted PDFs.
- **OpenAI** for enabling automated commentary summaries using GPT.
- **pdfbook2** for enabling easy conversion of PDFs into booklet format.
