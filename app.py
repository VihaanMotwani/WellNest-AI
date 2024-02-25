import io

import streamlit as st
import os
from io import BytesIO
import csv
import pandas as pd
import tiktoken
from langchain.chains.summarize import load_summarize_chain
import textwrap
from time import monotonic
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import PIL
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image as RLImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import openai
import tempfile
import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import dateutil.parser
from tqdm.auto import tqdm
import seaborn as sns
st.title("WellNest AI")
st.subheader("Empowering Organisations with AI-Driven Well-Being and Sentiment Analysis for a Positive Workplace Culture and to Boost Productivity.")

text_data = st.file_uploader(
    "Upload work environment rating text blocks csv with rows as 'dept' & 'feed'. The data limit is 300")
numeric_data = st.file_uploader("Upload work environment numeric data csv")

download_button_label = "Create Summary Report"

# Set up OpenAI API key
llm = ChatOpenAI(temperature=0, openai_api_key="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", model_name="gpt-3.5-turbo")
# Function for text summarization using GPT-3


if text_data is not None and numeric_data is not None:

    df1 = pd.read_csv(text_data)

    total_text = ''.join(df1['feed'].astype(str))

    model_name = "gpt-3.5-turbo"

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        model_name=model_name
    )

    texts = text_splitter.split_text(total_text)

    docs = [Document(page_content=t) for t in texts]
    print(len(docs))

    prompt_template = """Write a concise summary of the following:
    {total_text}"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["total_text"])


    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens


    num_tokens = num_tokens_from_string(total_text, "gpt-3.5-turbo")
    print(num_tokens)

    gpt_35_turbo_max_tokens = 4097
    verbose = True

    if num_tokens < 4096:
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt, verbose=verbose,
                                     document_variable_name="total_text")
    else:
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt,
                                     verbose=verbose, document_variable_name="total_text")

    start_time = monotonic()
    summary = chain.run(docs)

    print(f"Chain type: {chain.__class__.__name__}")
    print(f"Run time: {monotonic() - start_time}")
    print(f"Summary: {textwrap.fill(summary, width=100)}")


    # Read summary data from the CSV file
    summary_stats = pd.read_csv(numeric_data)


    def create_summary_report(summary, summary_stats, best_image_path):
        fileName = "summary_report.pdf"
        documentTitle = "summary_report"
        title = "Summary Report"
        styles = getSampleStyleSheet()
        body_style = styles["Normal"]
        if summary:
            summary_paragraph = Paragraph(summary, body_style)
        else:
            summary_paragraph = ['No summary available']

        packet = io.BytesIO()

        pdf = canvas.Canvas(packet)
        pdf.setTitle(documentTitle)
        pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))
        pdf.setFont('Arial', 24)
        pdf.drawCentredString(300, 770, title)
        pdf.line(30, 760, 550, 760)
        text = pdf.beginText(40, 600)
        text.setFont("Courier", 12)
        text.setFillColor(colors.black)
        summary_paragraph.wrapOn(pdf,500,400)
        summary_paragraph.drawOn(pdf, 40, 680)

        img = ImageReader(best_image_path)
        pdf.drawImage(img, x = -60, y = 100, width=700, height=200)

        # Example
        numerical_summary_table = Table([
            ("Mean", "Median", "Standard Deviation"),
            *(
                (summary_stats[col].mean(), summary_stats[col].median(), summary_stats[col].std())
                for col in summary_stats.columns
            ),
        ])
        numerical_summary_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('GRID', (0, 0), (-1, -1), 1, colors.gray),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        # Add the table to the PDF
        numerical_summary_table.wrapOn(pdf, 200, 400)  # Adjust width and height as needed
        numerical_summary_table.drawOn(pdf, 40, 400)
        pdf.save()
        packet.seek(0)
        return packet

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from transformers import (
            pipeline,
            AutoTokenizer,
            AutoModelForSequenceClassification
        )

    model_id = "cardiffnlp/twitter-roberta-base-sentiment"

    # load the model from huggingface
    model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=3
        )

    # load the tokenizer from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # load the tokenizer and model into a sentiment analysis pipeline
    nlp = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device
        )

    labels = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive"
        }

    # load the model from huggingface
    retriever = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            device=device
        )

    # connect to pinecone environment
    pc = Pinecone(api_key="adc052fa-fe00-40bd-ae1a-011f955b497d")
    index = pc.Index("sentiment-mining")


    def get_sentiment(feed):
        # pass the reviews through sentiment analysis pipeline
        sentiments = nlp(feed)
        # extract only the label and score from the result
        l = [labels[x["label"]] for x in sentiments]
        s = [x["score"] for x in sentiments]
        return l, s


        # we will use batches of 64
    batch_size = 64

    for i in tqdm(range(0, len(df1), batch_size)):
        # find end of batch
        i_end = min(i + batch_size, len(df1))
        # extract batch
        batch = df1.loc[i:i_end]
        # generate embeddings for batch
        emb = retriever.encode(batch["feed"].tolist()).tolist()
        # convert review_date to timestamp to enable period filters
        # timestamp = get_timestamp(batch["review_date"].tolist())
        # batch["timestamp"] = timestamp
        # get sentiment label and score for reviews in the batch
        label, score = get_sentiment(batch["feed"].tolist())
        batch["label"] = label
        batch["score"] = score
        # get metadata
        meta = batch.to_dict(orient="records")
        # create unique IDs
        ids = [f"{idx}" for idx in range(i, i_end)]
        # add all to upsert list
        to_upsert = list(zip(ids, emb, meta))
        # upsert/insert these records to pinecone
        _ = index.upsert(vectors=to_upsert)

        # check that we have all vectors in index
    index.describe_index_stats()


    query = "what is the level of collaboration in the company?"
    # generate dense vector embeddings for the query
    xq = retriever.encode(query).tolist()
    # query pinecone
    result = index.query(vector=xq, top_k=400, include_metadata=True)

    def count_sentiment(result):
        # store count of sentiment labels
        sentiments = {
            "negative": 0,
            "neutral": 0,
            "positive": 0,
        }
        # iterate through search results
        for r in result["matches"]:
            # extract the sentiment label and increase its count
            sentiments[r["metadata"]["label"]] += 1
        return sentiments


    sentiment = count_sentiment(result)

    # plot a barchart using seaborn
    sns.barplot(x=list(sentiment.keys()), y=list(sentiment.values()))

    depts = [
        "IT",
        "Operation/Manufacturing/Production",
        "Administration",
        "Finance & Accounting",
        "Marketing",
        "R&D",
        "Sales",
        "Logistics",
        "Customer Support",
        "Legal",
        ]

    queries = {
        "Diversity": "is the company engaging in diversity",
        # "Innovation": "innovation in the company",
        # "Creativity": "How is creativity supported by the amosphere?",
        # "Sustainability": "work environment commit to sustainbility",
        # "Collaboration": "collaboration in work"
    }

    dept_sentiments = []

    # iterate through the hotels
    for dept in depts:
        result = []
        # iterate through the keys and values in the queries dict
        for area, query in queries.items():
            # generate query embeddings
            xq = retriever.encode(query).tolist()
            # query pinecone with query embeddings and the hotel filter
            xc = index.query(vector=xq, top_k=500, include_metadata=True, filter={"dept": dept})
            # get an overall count of customer sentiment
            sentiment = count_sentiment(xc)
            # sort the sentiment to show area and each value side by side
            for k, v in sentiment.items():
                data = {
                    "area": area,
                    "label": k,
                    "value": v
                }
                # add the data to result list
                result.append(data)
        # convert the
        dept_sentiments.append({"dept": dept, "df": pd.DataFrame(result)})

    # create the figure and axes to plot barchart for all hotels
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(25, 4.5))
    plt.subplots_adjust(hspace=0.25)


    counter = 0
    # iterate through each hotel in the list and plot a barchart
    for d, ax in zip(dept_sentiments, axs.ravel()):
        # plot barchart for each hotel
        sns.barplot(x="label", y="value", hue="area", data=d["df"], ax=ax)
        # display the hotel names
        ax.set_title(d["dept"])
        # remove x labels
        ax.set_xlabel("")
        # remove legend from all charts except for the first one
        counter += 1
        if counter != 1: ax.get_legend().remove()
        # display the full figure
    plt.savefig("/Users/sparshjain/Downloads/best_sentiment.png")
    best_file = "/Users/sparshjain/Downloads/best_sentiment.png"
    plt.show()

    # pinecone.delete_index(sentiment-mining)
    with st.spinner("Generating PDF, please wait..."):
        st.markdown("### Summary Report")
        if st.button(download_button_label, key = "download_summary_report"):
            pdf = create_summary_report(summary, summary_stats, best_file)
            st.download_button(label="Download PDF", data=pdf, file_name='summary_report.pdf', mime="application/pdf")


else:
    st.warning("Please upload both text and numeric data files.")

