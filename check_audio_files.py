from gtts import gTTS

# Access granted message
tts1 = gTTS("Access granted")
tts1.save("access_granted.mp3")

# Locking message
tts2 = gTTS("The door is being locked")
tts2.save("lock_message.mp3")

print("Audio files created successfully!")
