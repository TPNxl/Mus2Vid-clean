from emotion import *
from features_modified import *
from genre_prediction import *
from image_generation import *
from prompting import *
from img_display_thread_amp import *
import time
import os

STARTING_CHUNK = 1024

new_image = False

def display_images_old(pipe):
    for i in range(len(pipe)):
        image = pipe[i]
        name = f"image_output_cache/image%d.png" % int(round(time.time() * 10, 1))
        image.save(name)

def main():
    dir = 'image_output_cache'
    if not os.path.exists(dir):
        os.mkdir(dir)
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    SPA_Thread = SinglePyAudioThread(name="SPA_Thread", starting_chunk_size=STARTING_CHUNK)
    MMF_Thread = ModifiedMIDIFeatureThread(name="MMF_Thread", SinglePyAudioThread=SPA_Thread)
    Emo_Thread = EmotionClassificationThreadSPA(name='Emo_Thread',
                                                SPA_Thread=SPA_Thread)
    GP_Thread = ModifiedGenrePredictorThread(name='GP_Thread',
                                             MF_Thread=MMF_Thread,
                                             SPA_Thread=SPA_Thread)

    Prompt_Thread = PromptGenerationThread(name='Prompt_Thread',
                                           genre_thread=GP_Thread,
                                           emotion_thread=Emo_Thread,
                                           audio_thread=SPA_Thread)

    Img_Thread = ImageGenerationThread(name='Img_Thread',
                                       Prompt_Thread=Prompt_Thread,
                                       display_func=None,
                                       audio_thread=SPA_Thread)
    Display_Thread = ImageDisplayThreadWithAmpTracking(name="Display_Thread",
                                         Prompt_Thread=Prompt_Thread,
                                         Img_Thread=Img_Thread,
                                         SPA_Thread=SPA_Thread)

    print("All threads init'ed")

    try:

        # Display_Thread.start()
        # print("============== Display started")
        SPA_Thread.start()
        print("============== SPA started")
        MMF_Thread.start()
        print("============== MMF started")
        Emo_Thread.start()
        print("============== Emo started")
        GP_Thread.start()
        print("============== GP started")
        Prompt_Thread.start()
        print("============== Prompt started")
        Img_Thread.start()
        print("============== Img started")
        start = time.time()
        while True:
            print("\n\n")
            print("Prompt: ", Prompt_Thread.prompt)
            # print("Buffer size: ", SPA_Thread.buffer_index)
            print(time.time() - start)
            print(SPA_Thread.input_on)
            if Emo_Thread.emo_values is not None:
                print(f"Valence: %.2f, Arousal: %.2f" % (Emo_Thread.emo_values[0], Emo_Thread.emo_values[1]))
            if GP_Thread.genre_output is not None:
                print("Genre output:", GP_Thread.genre_output)
            if Emo_Thread.emo_values is not None and GP_Thread.genre_output is not None and SPA_Thread.input_on and Prompt_Thread.prompt is not None and Prompt_Thread.prompt != "Black screen" and Prompt_Thread.prompt != "Blank screen":
                print(f"%.2f,%.2f,%.2f,%s,%s\n" % (time.time() - start, Emo_Thread.emo_values[0], Emo_Thread.emo_values[1], GP_Thread.genre_output, Prompt_Thread.prompt))
            time.sleep(1)
    except KeyboardInterrupt:
        SPA_Thread.stop_request = True
        MMF_Thread.stop_request = True
        GP_Thread.stop_request = True
        Emo_Thread.stop_request = True
        Prompt_Thread.stop_request = True
        Img_Thread.stop_request = True
        Display_Thread.stop_request = True
        

if __name__ == "__main__":
    main()