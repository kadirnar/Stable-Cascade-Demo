import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import gradio as gr


# Initialize the prior and decoder pipelines
prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to("cuda")
prior.enable_xformers_memory_efficient_attention()

decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=torch.float16).to("cuda")
decoder.enable_xformers_memory_efficient_attention()

def generate_images(
    prompt="a photo of a girl",
    negative_prompt="bad,ugly,deformed",
    height=1024,
    width=1024, 
    guidance_scale=4.0, 
    prior_inference_steps=20, 
    decoder_inference_steps=10
    ):
    """
    Generates images based on a given prompt using Stable Diffusion models on CUDA device.
    Parameters:
    - prompt (str): The prompt to generate images for.
    - negative_prompt (str): The negative prompt to guide image generation away from.
    - height (int): The height of the generated images.
    - width (int): The width of the generated images.
    - guidance_scale (float): The scale of guidance for the image generation.
    - prior_inference_steps (int): The number of inference steps for the prior model.
    - decoder_inference_steps (int): The number of inference steps for the decoder model.
    Returns:
    - List[PIL.Image]: A list of generated PIL Image objects.
    """

    # Generate image embeddings using the prior model
    prior_output = prior(
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        num_inference_steps=prior_inference_steps
    )

    # Generate images using the decoder model and the embeddings from the prior model
    decoder_output = decoder(
        image_embeddings=prior_output.image_embeddings.half(),
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,  # Guidance scale typically set to 0 for decoder as guidance is applied in the prior
        output_type="pil",
        num_inference_steps=decoder_inference_steps
    ).images

    return decoder_output


def web_demo():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                text2image_prompt = gr.Textbox(
                    lines=1,
                    placeholder="Prompt",
                    show_label=False,
                )

                text2image_negative_prompt = gr.Textbox(
                    lines=1,
                    placeholder="Negative Prompt",
                    show_label=False,
                )
                with gr.Row():
                    with gr.Column():
                        text2image_height = gr.Slider(
                            minimum=128,
                            maximum=1280,
                            step=32,
                            value=512,
                            label="Image Height",
                        )

                        text2image_width = gr.Slider(
                            minimum=128,
                            maximum=1280,
                            step=32,
                            value=512,
                            label="Image Width",
                        )
                        with gr.Row():
                            with gr.Column():
                                text2image_guidance_scale = gr.Slider(
                                    minimum=0.1,
                                    maximum=15,
                                    step=0.1,
                                    value=4.0,
                                    label="Guidance Scale",
                                )                
                                text2image_prior_inference_step = gr.Slider(
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    value=20,
                                    label="Prior Inference Step",
                                )                
                                
                                text2image_decoder_inference_step = gr.Slider(
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    value=10,
                                    label="Decoder Inference Step",
                                )               
                text2image_predict = gr.Button(value="Generate Image")
                
            with gr.Column():
                output_image = gr.Gallery(
                    label="Generated images",
                    show_label=False,
                    elem_id="gallery",
                )
                
            text2image_predict.click(
                fn=generate_images,
                inputs=[
                    text2image_prompt,
                    text2image_negative_prompt,
                    text2image_height,
                    text2image_width,
                    text2image_guidance_scale,
                    text2image_prior_inference_step,
                    text2image_decoder_inference_step
                ],
                outputs=output_image,
            )