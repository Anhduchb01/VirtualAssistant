import streamlit as st
from streamlit_chat import message as st_message
from transformers import AutoTokenizer, BlenderbotForConditionalGeneration ,MarianMTModel, MarianTokenizer
import torch
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import soundfile as sf
import queue
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import scipy.signal as signal
import pydub
import numpy as np
@st.cache_resource 
def get_models_chat():
	# it may be necessary for other frameworks to cache the model
	# seems pytorch keeps an internal state of the conversation
	model_name = "facebook/blenderbot-400M-distill"
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
	return tokenizer, model
@st.cache_resource 
def get_models_v_e():
	# it may be necessary for other frameworks to cache the model
	# seems pytorch keeps an internal state of the conversation
	model_name = "Helsinki-NLP/opus-mt-vi-en"
	model = MarianMTModel.from_pretrained(model_name)
	tokenizer = MarianTokenizer.from_pretrained(model_name)
	return tokenizer, model
@st.cache_resource
def get_models_e_v():
	# it may be necessary for other frameworks to cache the model
	# seems pytorch keeps an internal state of the conversation
	model_name = "Helsinki-NLP/opus-mt-en-vi"
	model = MarianMTModel.from_pretrained(model_name)
	tokenizer = MarianTokenizer.from_pretrained(model_name)
	return tokenizer, model
@st.cache_resource
def get_models_voice():
	# it may be necessary for other frameworks to cache the model
	# seems pytorch keeps an internal state of the conversation
	model_path = "../models/wav2vec2-vietnamese"
	processor = Wav2Vec2Processor.from_pretrained(model_path)
	model = Wav2Vec2ForCTC.from_pretrained(model_path)
	return processor, model



	
def predict_voice(file):
	speech, sample_rate = sf.read(file)
	print(sample_rate)

	print(speech)
	print(speech.shape)
	speech1 = speech[:, 0]
	print(speech1.shape)
	processor,model = get_models_voice()
	input_values = processor(speech1,sample_rate=sample_rate, return_tensors="pt", padding="longest").input_values  # Batch size 1

	# retrieve logits
	logits = model(input_values).logits

	# take argmax and decode
	predicted_ids = torch.argmax(logits, dim=-1)
	transcription = processor.batch_decode(predicted_ids)
	print(transcription)
	return transcription
def translate_vi_en(text):
		# Load the model and tokenizer
		tokenizer,model = get_models_v_e()
		# Tokenize the input text
		input_ids = tokenizer.encode(text, return_tensors="pt")

		# Translate the input text using the model
		translated = model.generate(input_ids)

		# Decode the translated output and print it
		output = tokenizer.decode(translated[0], skip_special_tokens=True)
		return output
def translate_en_vi(text):
		tokenizer,model = get_models_e_v()
		# Tokenize the input text
		input_ids = tokenizer.encode(text, return_tensors="pt")
		# Translate the input text using the model
		translated = model.generate(input_ids)
		# Decode the translated output and print it
		output = tokenizer.decode(translated[0], skip_special_tokens=True)
		return output

def predict(text):
	tokenizer, model = get_models_chat()
	user_message_en = translate_vi_en(text)
	inputs = tokenizer(user_message_en, return_tensors="pt")
	result = model.generate(**inputs)
	message_bot = tokenizer.decode(
		result[0], skip_special_tokens=True
	)  # .replace("<s>", "").replace("</s>", "")
	message_bot_vn = translate_en_vi(message_bot)
	print('message_bot_vn :',message_bot_vn)
	st.session_state.history.append({"message": text, "is_user": True})
	st.session_state.history.append({"message": message_bot_vn, "is_user": False})
	st.session_state.input_text = ""
def  generate_answer():
	
	user_message = st.session_state.input_text
	print('user_message :',user_message.lower())
	predict(user_message)

	
	

def main():
	webrtc_ctx = webrtc_streamer(
		key="sendonly-audio",
		mode=WebRtcMode.SENDONLY,
		audio_receiver_size=256,
		client_settings=ClientSettings(
			rtc_configuration={
				"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
			},
			media_stream_constraints={
				"audio": True,
			},
		),
	)

	if "audio_buffer" not in st.session_state:
		st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

	status_indicator = st.empty()

	while True:
		if webrtc_ctx.audio_receiver:
			try:
				audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
			except queue.Empty:
				status_indicator.write("No frame arrived.")
				continue

			status_indicator.write("Running. Say something!")

			sound_chunk = pydub.AudioSegment.empty()
			for audio_frame in audio_frames:
				sound = pydub.AudioSegment(
					data=audio_frame.to_ndarray().tobytes(),
					sample_width=audio_frame.format.bytes,
					frame_rate=audio_frame.sample_rate,
					channels=len(audio_frame.layout.channels),
				)
				sound_chunk += sound

			if len(sound_chunk) > 0:
				st.session_state["audio_buffer"] += sound_chunk
		else:
			status_indicator.write("AudioReciver is not set. Abort.")
			break

	audio_buffer = st.session_state["audio_buffer"]
	
	if not webrtc_ctx.state.playing and len(audio_buffer) > 0:
		st.info("Writing wav to disk")
		print(audio_buffer)
		
		audio_buffer.export("temp.wav", format="wav")
		speech, sample_rate = sf.read('./temp.wav')
		print(sample_rate)

		print(speech)
		
		resampled_audio = signal.resample(speech, int(len(speech) * 16000 / sample_rate))
		print(resampled_audio)
		# speech1 = resampled_audio.reshape(-1)
		sf.write("./temp1.wav", resampled_audio, 16000)
		
		text =predict_voice('./temp1.wav')
		print('text',text[0])
		predict(text[0])

		# Reset
		st.session_state["audio_buffer"] = pydub.AudioSegment.empty()



if __name__ == "__main__":
	if "history" not in st.session_state:
		st.session_state.history = []
	st.title("Hello Chatbot")
	main()
	
	st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

	for chat in st.session_state.history:
		st_message(**chat)  # unpacking