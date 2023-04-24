#%%
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY") #
text = """ We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation. Using our efficient model in a data collection loop, we built the largest segmentation dataset to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images. The model is designed and trained to be promptable, so it can transfer zero-shot to new image distributions and tasks. We evaluate its capabilities on numerous tasks and find that its zero-shot performance is impressive -- often competitive with or even superior to prior fully supervised results. We are releasing the Segment Anything Model (SAM) and corresponding dataset (SA-1B) of 1B masks and 11M images at this https URL to foster research into foundation models for computer vision. """

text = """This paper proposes a method for generating images of customized objects specified by users. The method is based on a general framework that bypasses the lengthy optimization required by previous approaches, which often employ a per-object optimization paradigm. Our framework adopts an encoder to capture high-level identifiable semantics of objects, producing an object-specific embedding with only a single feed-forward pass. The acquired object embedding is then passed to a text-to-image synthesis model for subsequent generation. To effectively blend a object-aware embedding space into a well developed text-to-image model under the same generation context, we investigate different network designs and training strategies, and propose a simple yet effective regularized joint training scheme with an object identity preservation loss. Additionally, we propose a caption generation scheme that become a critical piece in fostering object specific embedding faithfully reflected into the generation process, while keeping control and editing abilities. Once trained, the network is able to produce diverse content and styles, conditioned on both texts and objects. We demonstrate through experiments that our proposed method is able to synthesize images with compelling output quality, appearance diversity, and object fidelity, without the need of test-time optimization. Systematic studies are also conducted to analyze our models, providing insights for future work."""
response = openai.Embedding.create(
  model="text-embedding-ada-002",
  input=text
)
embeddings2 = response['data'][0]['embedding']

#%%
complete_resp = openai.Completion.create(
  # model="text-davinci-003",
  model="text-curie:001",
  prompt="Abstract\n"+text+"\nMain Text\n",
  max_tokens=256,
  temperature=0.5
)
#%%

# with textwrap
import textwrap
print(textwrap.fill(complete_resp['choices'][0]['text'], 80))

