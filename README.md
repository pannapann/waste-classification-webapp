# Waste Classification Web App

This repository contains the code for a waste classification web application built with Streamlit. The application allows users to take a photo using a webcam connected to a Khadas VIM2 and classify the type of waste into 5 categories: cardboard, glass, metal, paper, and plastic.

## Getting Started

To set up and run the web application, follow these steps:

### Prerequisites

- Python 3.7+
- Streamlit
- TensorFlow
- OpenCV
- PIL (Pillow)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/your_username/waste-classification-webapp.git
```

2. Navigate to the project directory:
```bash
cd waste-classification-webapp
```
3. Install the required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

4. Place your trained models (model.h5, model.tflite, model_pruned.tflite, model_quantized.tflite, model_pruned_quantized.tflite) in the project directory.

### Usage

1. Start the Streamlit web application:
```bash
streamlit run app.py
```

2. Open the web application in your browser using the URL provided in the terminal.

3. Choose the desired model from the dropdown menu.

4. Take a photo using the webcam connected to the Khadas VIM2.

5. The application will classify the waste in the image into one of the 5 categories: cardboard, glass, metal, paper, or plastic.



## Acknowledgments

- [OpenAI](https://www.openai.com/) for the GPT model that helped in generating parts of this README.
- [Streamlit](https://streamlit.io/) for the awesome web app framework.
- [TensorFlow](https://www.tensorflow.org/) for the machine learning library used in this project.
- [OpenCV](https://opencv.org/) for the image processing library used in this project.





