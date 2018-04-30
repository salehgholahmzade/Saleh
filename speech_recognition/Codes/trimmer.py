from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

def trimmer(address_in ):
    sound_file = AudioSegment.from_wav(address_in)
    chunks = split_on_silence(sound_file, min_silence_len=100, silence_thresh= -40)
        # length of silence in ms

        # consider it silent if quieter than -60 dBFS



    full_path = os.path.join(os.getcwd() , 'Trimmed_voice')
    transcript_filename = 'saleh'
    for i, chunk in enumerate(chunks):
        chunk.export(os.path.join(full_path, transcript_filename + "{}.wav".format("%02d" % i)), format="wav")


# if __name__ == '__main__'  :
#
#     trimmer('tahour.wav' )
