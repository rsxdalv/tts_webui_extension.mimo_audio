import gradio as gr


def mimo_audio_ui():
    gr.Markdown(
        """
    # Mimo audio
    
    This is a template extension. Replace this content with your extension's functionality.
    
    To use it, simply modify this UI and add your custom logic.
    """
    )
    
    # Add your UI components here
    # Example:
    # with gr.Row():
    #     with gr.Column():
    #         input_text = gr.Textbox(label="Input")
    #         button = gr.Button("Process")
    #     with gr.Column():
    #         output_text = gr.Textbox(label="Output")
    # 
    # button.click(
    #     fn=your_processing_function,
    #     inputs=[input_text],
    #     outputs=[output_text],
    #     api_name="mimo_audio",
    # )


def extension__tts_generation_webui():
    mimo_audio_ui()
    
    return {
        "package_name": "tts_webui_extension.mimo_audio",
        "name": "Mimo audio",
        "requirements": "git+https://github.com/rsxdalv/tts_webui_extension.mimo_audio@main",
        "description": "A template extension for TTS Generation WebUI",
        "extension_type": "interface",
        "extension_class": "text-to-speech",
        "author": "Your Name",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/rsxdalv/tts_webui_extension.mimo_audio",
        "extension_website": "https://github.com/rsxdalv/tts_webui_extension.mimo_audio",
        "extension_platform_version": "0.0.1",
    }


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        with gr.Tab("Mimo audio", id="mimo_audio"):
            mimo_audio_ui()

    demo.launch(
        server_port=7772,  # Change this port if needed
    )
