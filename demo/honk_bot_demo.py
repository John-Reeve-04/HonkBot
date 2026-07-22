"""
Demo info
PLAN: have a few functions:
    Loudest Tone (Done)
        Will be just printing what the loudest tone is.

    Most Likely Ship (Done)
        Will be printing if there's a ship of a certain length
        Will be ust a rough idea to get it working for my presentation

    Be Loud! (Done)
        Will take in a loudness multiplier
        and what the audio average for a room for a few seconds,
        then you have to show you can be louder than the threshold (average*multiplier)


First function shows just general functionality,
second shows a vague theory of what I want to do,
third is fun and shows more theory of how to filter out unwanted or background audio
"""

# libraries

import sys

from collections import deque

from time import sleep

# for audio capture
import pyaudio

# for audio analysis
import scipy

# for conversion of audio to analysis
import numpy

# any lists, structures and variables for the code

# chunk/rate = recording time (seconds), currently splits into 42 chunks per second
CHUNK = 1050

# our recording format, this one works well
FORMAT = pyaudio.paInt16

# mono audio to get correct number of samples
CHANNELS = 1

# sample rate
RATE = 44100

# initialise audio code
P = pyaudio.PyAudio()

# open recording stream
STREAM = P.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# using a sliding window of 1 second that ensures honks don't get lost between seconds
WINDOW = deque(maxlen=42)

TRY_AGAIN = "Run again (1), or go back to main menu (2)? "

AGAIN_YES = "Got it! Will clear and run again...\n"

AGAIN_NO = "Got it! Returning to main menu...\n"


def main():
    """
    Start function to launch the demo functions
    """
    func = 0
    print("""Hello! Welcome to my demo!!!
            We have 3 little things for you to try. Those being...
            Loudest Tone
            Most Likely Ship
            & Be Loud!!!""")
    while func == 0:
        func = int(input("Which would you like to try? 1, 2 or 3? Or 4 to quit! "))
        print("\n")

        if func == 1:
            loudest_tone()

        elif func == 2:
            most_likely_ship()

        elif func == 3:
            be_loud()

        elif func == 4:
            print("Thanks for trying my demo! ^_^")
            STREAM.stop_stream()
            STREAM.close()
            P.terminate()
            sleep(0.5)
            sys.exit(0)

        else:
            print("I did not recognise that sorry, please try again.\n")
            sleep(1)


def loudest_tone():
    """
    The loudest tone function

    Just made to show the general idea of capturing specific frequencies.

    Takes 1 second (42 samples) of audio,
    then gets the maximum magnitude frequency from it.
    """
    print("Taking a second of audio...")
    sleep(0.5)
    record_second()

    print("Got a second! Will display loudest tone in a moment...\n")
    sleep(0.5)
    # now get the audio info
    fft_array = fft()
    magnitudes = get_db_mags(numpy.abs(fft_array), True)

    print("Loudest frequency in the last second was:", magnitudes.argmax())
    max_mag = magnitudes[magnitudes.argmax()]
    print("It registered a magnitude (relative dB, 0 is max) of:", round(max_mag, 2))
    sleep(1)

    WINDOW.clear()
    run_again(1)


def most_likely_ship():
    """
    The most likely ship function

    Made to show a vague idea for how to identify a ship of a specific size compared to others.

    NOTE: This is demo code, not the final implementation

    Repeatedly takes 1 second of audio,
    analyses the specific bins for the different ship sizes,
    compares them to see if any are a threshold above the others,
    and thus has a "boat horn" present!
    """
    print(
        "I will start recording audio now "
        "and check each second for if a ship horn noise has been detected."
    )
    while True:
        found_ship = True  # defaults to true, flip later, keeps it short
        record_second()

        fft_array = fft()
        magnitudes = get_db_mags(numpy.abs(fft_array), False)

        # large ships from 70-200Hz, medium ships from 130-350Hz and small from 250-700Hz
        large_avg = get_top3_avg(magnitudes, 70, 201)
        medium_avg = get_top3_avg(magnitudes, 130, 351)
        small_avg = get_top3_avg(magnitudes, 250, 701)

        WINDOW.clear()  # to be ready for next second of audio

        # NOTE: These values are tuned for my specific demo location and hardware

        if (
            (small_avg > (large_avg * 1.8))
            and (small_avg > (medium_avg * 1.8))
            and small_avg > 50
        ):
            print("That might be a small boat!")

        elif (
            (medium_avg > (small_avg * 1.8))
            and (medium_avg > (large_avg * 1.8))
            and medium_avg > 35
        ):
            print("That might be a medium boat!")

        elif (
            (large_avg > (medium_avg * 1.5))
            and (large_avg > (small_avg * 1.8))
            and large_avg > 20
        ):
            print("That might be a large boat!")

        elif (
            (large_avg > (small_avg * 1.8))
            and (medium_avg > (small_avg * 1.8))
            and medium_avg > 35
            and large_avg > 30
        ):  # account for overlap
            print("The horn is in an overlap area, that might be a large or medium boat!")

        elif (
            (medium_avg > (large_avg * 2))
            and (small_avg > (large_avg * 2))
            and medium_avg > 35
            and small_avg > 35
        ):  # account for overlap
            print("The horn is in an overlap area, that might be a medium or small boat!")

        else:
            print("No boat found. Clearing audio and trying again...")
            found_ship = False
            sleep(1)

        # ask to run again if a ship honk was found (didn't hit else above)
        if found_ship:
            run_again(2)


def be_loud():
    """
    Be loud!
    A fun little demo function where you get to yell at my laptop
    and see if you can be louder than the environment.

    Shows some theory on how to filter out background audio,
    how we can make it just focus on noises that are significantly louder.

    It asks for how loud you think you can be,
    gets 5 seconds of audio and the average magnitudes from that to serve as a baseline,
    then listens more until the input reaches the strength you chose.
    """
    loudness = int(
        input("How loud can you be? Gimme a number, a multiple compared to your environment. ")
    )
    print("Will set the threshold levels in just a moment...")
    sleep(1)

    print("I will get 5 seconds of audio to serve as a baseline.")
    print("It will take around 8 seconds to get, in order to get good samples.")
    print("Be as loud or quiet as you want in this time.")
    input("Press enter when ready.")
    # waits for input so there's time to read

    averages = []

    print("Starting in 3...")
    sleep(1)
    print("2..")
    sleep(1)
    print("1.")
    sleep(1)
    print("Capturing audio now!")

    while len(averages) != 5:
        record_second()
        fft_array = fft()
        magnitudes = numpy.abs(
            fft_array
        )  # using the weird millions of magnitude, since it works better for this function
        average = numpy.mean(magnitudes)
        averages.append(average)
        WINDOW.clear()
        sleep(0.5)

    print("Got it!!! Finding baseline volume now...")

    base_line = numpy.mean(averages)

    # NOTE: this is again, tuned for my specific demo location and hardware
    target_loudness = (
        base_line * loudness * 80
    )  # scales with loudness selected, needs a bit extra because the numbers are weird

    # start looking for loudness
    attempt = 0
    print(
        "Alright, let's start getting that audio! You need to be",
        loudness,
        "times louder than the audio from before!!!",
    )
    while attempt < target_loudness:
        # can be 5 seperate lines, but one line is cool
        attempt = numpy.max(
            numpy.abs(scipy.fft.rfft(numpy.frombuffer(STREAM.read(CHUNK), dtype=numpy.int16)))
        )

    print("AH!!! TOO LOUD!!! YOU DID IT!!! I GIVE UP!!! (ㅠ﹏ㅠ)")
    sleep(1)

    times_over = attempt / target_loudness

    print("You were", round(times_over, 2), "times louder than you needed to be, that was crazy!")
    sleep(2)

    run_again(3)


def run_again(loc):
    """
    Function to ask if the user wants to try the module they're in again
    Knows the module they're in from the loc number
    Have this here so I don't have to repeat the code
    """
    # get choice to continue or not
    choice = 0
    while choice not in (1, 2):
        choice = int(input(TRY_AGAIN))
    if choice == 1:
        print(AGAIN_YES)
        sleep(1)
        if loc == 1:
            loudest_tone()
        elif loc == 2:
            most_likely_ship()
        elif loc == 3:
            be_loud()
        else:
            print("Are you implementing a new method? Remember to add it to run_again!")
            print("Returning to main menu... \n")
            main()
    elif choice == 2:
        print(AGAIN_NO)
        sleep(1)
        main()


def get_db_mags(magnitudes, b):
    """
    Function to "normalise" our magnitudes to be "relative db"
    Have this here so I don't have to repeat the code
    """
    normalised_mags = magnitudes / 44100  # divide by number of samples
    normalised_mags = numpy.maximum(
        normalised_mags, 1e-10
    )  # make sure we have no zeros, replace them with a tiny number
    normalised_mags[1:] *= 2

    if b:
        normalised_mags /= 32768  # makes it from about -90 to 0

    db_mags = 20 * numpy.log10(normalised_mags)

    return db_mags


def record_second():
    """
    A snippet to record a chunk of audio and add it to our WINDOW until we have 1 seconds worth
    Have this here so I don't have to repeat the code
    """
    while len(WINDOW) != 42:
        audio = STREAM.read(CHUNK)
        data = numpy.frombuffer(audio, dtype=numpy.int16)
        WINDOW.append(data)


def fft():
    """
    Makes our fft array from the second of audio recorded
    Have this here so I don't have to repeat the code
    """
    this_second = numpy.concatenate(list(WINDOW))
    return scipy.fft.rfft(this_second)


def get_top3_avg(magnitudes, low, high):
    """
    Gets the average of the top 3 of a given ship range
    For the most_likely_ship function
    """
    ship_bin = magnitudes[low:high]
    top3 = numpy.partition(ship_bin, -3)[-3:]
    return numpy.mean(top3)


# run the program :)
if __name__ == "__main__":
    main()
