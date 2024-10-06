# Parasha Booklet Generator

This repository contains a Snakemake workflow that automates the creation of bilingual booklets for weekly Torah readings (parashot). The workflow fetches the text of each parasha in both Hebrew and English from the Sefaria API, formats the text into a PDF booklet, and generates the final output in a ready-to-print format.

## Features

- **Data Collection**: Automatically downloads the list of parashot from Sefaria and fetches the Hebrew and English texts for each portion.
- **Bilingual PDF Generation**: Generates PDFs with Hebrew and English side-by-side, using ReportLab to ensure proper text alignment and styling.
- **Text Wrapping and Alignment**: Correctly handles Hebrew text direction and alignment with `bidi` and `arabic_reshaper` libraries.
- **Booklet Format**: Uses `pdfbook2` to generate a ready-to-print booklet format from the generated PDF.

## Requirements

- **Python 3.x**
- **Snakemake**
- **ReportLab**: For PDF generation.
- **Requests**: For API requests.
- **BeautifulSoup4**: For HTML parsing.
- **bidi** and **arabic_reshaper**: For handling Hebrew text alignment.
- **pdfbook2**: For converting PDFs into booklet format.

### Python Libraries

You can install the required Python libraries using:

```sh
pip install snakemake reportlab requests beautifulsoup4 python-bidi arabic-reshaper
```

## Usage

1. **Clone the Repository**:

   ```sh
   git clone https://github.com/yourusername/parasha-booklet-generator.git
   cd parasha-booklet-generator
   ```
2. **Run the Workflow**:
   Use Snakemake to run the entire workflow and generate booklets for all parashot.

   ```sh
   snakemake -j1 all
   ```

   The `-c1` flag indicates that only one job will run at a time. Increase this number if you want parallel execution.
3. **Generated Output**:

   - The generated PDF booklets are stored in the `booklets/` directory.

## Workflow Overview

- **Checkpoint: `get_parashot_csv`**
  - Downloads the list of parashot from Sefaria.
- **Rule: `get_parasha`**
  - Fetches Hebrew and English texts for a given parasha from the Sefaria API.
- **Rule: `make_pdf`**
  - Generates a bilingual PDF for the parasha, with proper alignment of Hebrew and English texts.
- **Rule: `make_book`**
  - Converts the generated PDF into a ready-to-print booklet format using `pdfbook2`.

## Key Features and Implementation Details

- **Hebrew and English Text Handling**: The workflow makes use of `bidi` and `arabic_reshaper` to ensure that Hebrew text is rendered correctly and wraps appropriately in multi-line paragraphs.
- **Page Layout**: The generated PDFs include headers, verse numbers, and formatted Hebrew and English text side-by-side for easy reading.
- **Blank Page Addition**: Adds an extra blank page at the end of each PDF to facilitate printing as a booklet.

## Customization

- **Fonts**: The workflow uses `NotoSansHebrew` for Hebrew text. You can change the font by modifying the path in the `make_pdf` rule.
- **Translations**: The Sefaria API provides multiple translations. The workflow currently uses a specific translation for English text, but you can modify it by changing the `version` parameter in the URL.

## Potential Improvements

- **Error Handling**: Implement better error handling for network failures and missing data to make the workflow more robust.
- **Parallel Execution**: Increase the number of parallel jobs (`-j`) to speed up the processing of multiple parashot.
- **Configuration File**: Introduce a configuration file to make font paths, translation options, and other parameters easily customizable.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for suggestions or bug reports.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- **Sefaria** for providing access to their API and the text data used in this project.
- **ReportLab** for providing the tools to generate well-formatted PDFs.
- **pdfbook2** for enabling easy conversion of PDFs into booklet format.
