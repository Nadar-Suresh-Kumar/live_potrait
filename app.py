import os
import requests
import replicate
from langchain.chains import LLMChain
from langchain_community.llms import Replicate as LangChainReplicate
from langchain_core.prompts import PromptTemplate
import streamlit as st

# Streamlit app
st.title("Live Portrait Generation with Replicate API")

# Input box for the Replicate API token
api_token = st.text_input("Enter your Replicate API Token", type="password")

if api_token:
    # Set the Replicate API token
    os.environ["REPLICATE_API_TOKEN"] = api_token
      api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

    # Input fields for the parameters
    face_image_url = st.text_input("Face Image URL", value="https://replicate.delivery/pbxt/L0gy7uyLE5UP0uz12cndDdSOIgw5R3rV5N6G2pbt7kEK9dCr/0_3.webp")
    driving_video_url = st.text_input("Driving Video URL", value="https://replicate.delivery/pbxt/LEQxLFMUNZMiKt5PWjyMJIbTdvKAb5j3f0spuiEwt9TEbo8B/d0.mp4")

    if st.button("Generate Live Portrait"):
        # Define the LangChain prompt template
        template = """
        Create a detailed and creative live portrait description based on the following face image and driving video URLs:
        Face Image: {face_image}
        Driving Video: {driving_video}
        """
        prompt = PromptTemplate(template=template, input_variables=["face_image", "driving_video"])

        # Initialize the LangChain model
        llm = LangChainReplicate(
            model="meta/meta-llama-3-8b-instruct",
            model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
        )
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        # Generate the prompt using LangChain
        generated_prompt = llm_chain.run({
            "face_image": face_image_url,
            "driving_video": driving_video_url
        })

        # Log the generated prompt (if needed)


        # Define the input for the Replicate API
        input_data = {
            "face_image": face_image_url,
            "driving_video": driving_video_url
        }

        # Run the replicate model
        output = replicate.run(
            "fofr/live-portrait:067dd98cc3e5cb396c4a9efb4bba3eec6c4a9d271211325c477518fc6485e146",
            input=input_data
        )

        # Display the generated live portrait video
        st.write("Generated Live Portrait:")
        for video_url in output:
            st.video(video_url)

# Note: Make sure you have all necessary imports and dependencies installed
