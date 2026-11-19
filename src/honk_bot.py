"""
Filename: honk_bot.py
Author: John Reeve
Date: 26/04/2026 - 12/10/2026
Version: 0.3.0
Description: My code for my ENGR489 project. Split into classes.
License: GNU GENERAL PUBLIC LICENSE, Version 3 (GPL V3)
"""

# libraries

# to avoid buffer overflows in audio detection while posting to social media
import threading
import queue

# for time stuff
import time

# to post calendar time
from datetime import datetime

# for enviornment variables
import os
from dotenv import load_dotenv

# for audio capture
import pyaudio

# for audio analysis
import scipy

# for conversion of audio to analyse
import numpy

# for Bluesky
from atproto import Client, Request

# and to time out if a request gets stuck
from httpx import Timeout

# our recording variables
# chunk/rate = recording time (seconds), currently splits into 10 chunks per second
# (any more is too much for the raspberry pi to handle presently,
# would need to optimise code if we want/need more, which may or may not be possible)
CHUNK = 4800
# a recording type, works fine
FORMAT = pyaudio.paInt16
# mono audio
CHANNELS = 1
# sample rate (has to be 48000 to work with the mic we have on the Pi)
RATE = 48000

# where our horns should be
# https://www.ecolregs.com/index.php?Itemid=501&id=120&lang=en&option=com_k2&view=item, section b

# see NOTE: overlap in #12
# see NOTE: very small boats in #12

SML_HORN_LOW = 250
SML_HORN_HIGH = 700

MED_HORN_LOW = 130
MED_HORN_HIGH = 350

LRG_HORN_LOW = 70
LRG_HORN_HIGH = 200

# our minimum energy (loudness) to be a honk
MIN_MAGS = 90

# min time for a single blast (really 1 second, but needs to account for human error)
MIN_BLAST_TIME = 0.8

# max time between blasts before we stop counting it as a honk
# see NOTE: MAX_BLAST_GAP in #12
MAX_BLAST_GAP = 2.5

# our leeway variable maximums
BASE_DIFF_HORN_LEEWAY = 3
BASE_BLAST_LEEWAY = 3

# see NOTE: percent average in #12
SAMPLE_PERCENT = 0.2174


class HonkState:
    """
    To store three boolean variables which need to retain their state throughout the code.
    And avoid them being global!
    Could be unoptimal, but it works and was where I got up to by hand-in.
    """

    # for 2 prolonged blasts, see note in #32
    two_second_gap = False

    # for if we get overtake request
    heard_overtake = False

    # for if we've heard a blind corner call
    heard_blind = False


class HonkDetector:
    """Class to detect honks"""

    def __init__(self):
        """Initialise audio code"""
        # put first to avoid crashing on stream.read(CHUNK)
        self.poster = HonkPoster()  # to post about our honks
        self.decipherer = HonkDecipherer()  # to decipher honks

        self.p = pyaudio.PyAudio()

        # open recording stream
        self.stream = self.p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
        )

        # figure out frequency bins for different ship sizes before starting
        freqs = numpy.fft.rfftfreq(CHUNK, 1.0 / RATE)

        self.small_horn_bins = numpy.nonzero((freqs >= SML_HORN_LOW) & (freqs <= SML_HORN_HIGH))
        self.medium_horn_bins = numpy.nonzero((freqs >= MED_HORN_LOW) & (freqs <= MED_HORN_HIGH))
        self.large_horn_bins = numpy.nonzero((freqs >= LRG_HORN_LOW) & (freqs <= LRG_HORN_HIGH))

        # and all bins (minimum over 50 because anything under is too noisy)
        self.all_bins = numpy.nonzero(freqs > 50)

    def detector_main(self):
        """Our main code to do our whole process, loops forever"""
        # NOTE: honk = whole noise, blast = each noise the horn makes that combines to a signal

        honk_ongoing = False  # if our honk (multiple blasts) has started

        ship_size = None  # ship size

        start_honk_time = None  # start time of the honk
        start_blast_time = None  # start time of a single blast
        last_blast_end = None  # end time of last blast
        blast_ongoing = False  # if a blast is ongoing
        blasts_in_honk = []  # a collection of blasts (short/long) found, to be decoded

        # need these for special cases, see notes in #12
        first_blast_time = None
        diff_horn_leeway = BASE_DIFF_HORN_LEEWAY
        temp_ship_size = None
        temp_horn_start = None

        # how many samples can come up empty before a blast is ended, see note in #12
        blast_leeway = BASE_BLAST_LEEWAY

        # to check if 2 prolonged blasts have a 2 seocnd gap, see note in #32
        two_prolonged_timer = None

        # starts detection loop!
        # see NOTE: complexity in #12
        while True:
            horn_found = self.find_horns()
            now = time.time()

            # if we find horn noise
            if horn_found != "":

                # reset leeway
                blast_leeway = BASE_BLAST_LEEWAY

                # dealing with ship size

                # no ship type means new honk, set ship size
                if ship_size is None:
                    ship_size = horn_found

                # if we've already found a ship_size and this is different,
                # could be a response signal starting
                elif horn_found != ship_size:

                    # already started leeway in case of noise triggering this,
                    # check if size has changed again
                    if temp_ship_size is not None:
                        if horn_found != temp_ship_size:
                            # size changed again, reset countdown (and then other variables)
                            diff_horn_leeway = BASE_DIFF_HORN_LEEWAY

                    # want to give some leeway in case of noise
                    if diff_horn_leeway == BASE_DIFF_HORN_LEEWAY:
                        temp_horn_start = now  # in case new ship honk, want correct start time
                        temp_ship_size = horn_found

                    # reduce leeway
                    diff_horn_leeway -= 1

                    # if run out of leeway (likely not noise), post old honk
                    if diff_horn_leeway == 0:
                        # quickly check if we are dealing with a special case
                        if HonkState.two_second_gap and len(blasts_in_honk) != 2:
                            HonkState.two_second_gap = False

                        # if only one blast, want that time
                        if len(blasts_in_honk) == 1:
                            # (blast_time could have been messed up by bad blast after first)
                            honk_time = round(first_blast_time, 2)
                        else:
                            honk_time = round(now - start_honk_time, 2)

                        self.poster.socials_queue.put(
                            (
                                ship_size,
                                start_honk_time,
                                honk_time,
                                blasts_in_honk,
                                self.decipherer,
                            )
                        )

                        # and reset with temp variables so info is not lost
                        ship_size = horn_found
                        honk_ongoing = True
                        blast_ongoing = True
                        start_honk_time = temp_horn_start
                        start_blast_time = temp_horn_start
                        last_blast_end = None
                        blasts_in_honk = []
                        two_prolonged_timer = None
                        first_blast_time = None
                        temp_ship_size = None
                        # see NOTE: ship size changes in #12

                # horn found same as before,
                else:
                    # make sure diff_horn vars are reset
                    diff_horn_leeway = BASE_DIFF_HORN_LEEWAY
                    temp_ship_size = None

                # dealing with the honk

                # new honk
                if not honk_ongoing:
                    honk_ongoing = True
                    start_honk_time = now

                # honk ongoing (or just started), new blast
                if not blast_ongoing:
                    blast_ongoing = True
                    start_blast_time = now

            # either honk/blast is done, or a horn noise wasn't found

            # if a honk has started
            elif honk_ongoing:

                # if we have a blast ongoing
                if blast_ongoing:
                    # reduce leeway until 0
                    blast_leeway -= 1

                    if blast_leeway == 0:
                        # x samples had no more horn sounds, blast over
                        blast_ongoing = False
                        blast_time = now - start_blast_time

                        # only count if blast over the minimum, otherwise noise
                        if blast_time > MIN_BLAST_TIME:

                            # short blast (.7 seconds room of error)
                            if blast_time <= 1.5:
                                blasts_in_honk.append("short")
                                last_blast_end = now
                                if len(blasts_in_honk) == 1:  # first blast
                                    first_blast_time = blast_time
                                blast_leeway = BASE_BLAST_LEEWAY

                            # long blast (.5 seconds room of error on either side)
                            elif 3.5 <= blast_time <= 6.5:
                                if two_prolonged_timer is None:
                                    two_prolonged_timer = now
                                elif now - two_prolonged_timer >= 2:
                                    HonkState.two_second_gap = True

                                blasts_in_honk.append("long")
                                last_blast_end = now
                                if len(blasts_in_honk) == 1:  # first blast
                                    first_blast_time = blast_time
                                blast_leeway = BASE_BLAST_LEEWAY

                            # between blast lengths
                            # seems to be the length of most ferry departure horns in Wellington
                            elif 1.5 < blast_time < 3.5:
                                # see NOTE: ferry horns in #27

                                # if no blasts found, could be single departure blast
                                if not blasts_in_honk:
                                    if ship_size == "medium":
                                        blasts_in_honk.append("departure")
                                        self.poster.socials_queue.put(
                                            (
                                                ship_size,
                                                start_honk_time,
                                                round(blast_time, 2),
                                                blasts_in_honk,
                                                self.decipherer,
                                            )
                                        )

                                    # reset possibly changed variables either
                                    # for next loop or because bad blast
                                    ship_size = None
                                    honk_ongoing = False
                                    start_honk_time = None
                                    start_blast_time = None
                                    blast_ongoing = False
                                    blasts_in_honk = []  # might have "departure" in it
                                    blast_leeway = BASE_BLAST_LEEWAY

                                # partial time reset, as there has been blasts found already
                                # (want to keep good info already found)
                                else:
                                    start_blast_time = None
                                    blast_ongoing = False
                                    blast_leeway = BASE_BLAST_LEEWAY

                            # way too long, might be distress signal
                            elif blast_time > 8.0:
                                blasts_in_honk = []  # reset, only care about this blast
                                HonkState.two_second_gap = False  # also reset

                                blasts_in_honk.append("distress")

                                self.poster.socials_queue.put(
                                    (
                                        ship_size,
                                        start_honk_time,
                                        round(blast_time, 2),
                                        blasts_in_honk,
                                        self.decipherer,
                                    )
                                )

                                # and reset variables for next loop
                                ship_size = None
                                honk_ongoing = False
                                start_honk_time = None
                                start_blast_time = None
                                last_blast_end = None
                                blast_ongoing = False
                                blasts_in_honk = []
                                two_prolonged_timer = None
                                first_blast_time = None
                                blast_leeway = BASE_BLAST_LEEWAY

                            # this is between 6.5 - 8 seconds, unknown noise, reset
                            else:
                                # no blasts found yet, nothing to lose, full reset
                                if not blasts_in_honk:
                                    ship_size = None
                                    honk_ongoing = False
                                    start_honk_time = None
                                    start_blast_time = None
                                    blast_ongoing = False
                                    blast_leeway = BASE_BLAST_LEEWAY

                                # partial time reset
                                else:
                                    start_blast_time = None
                                    blast_ongoing = False
                                    blast_leeway = BASE_BLAST_LEEWAY

                        # horn was too short, < 0.8 seconds, noise, do a full or partial reset
                        else:
                            if not blasts_in_honk:
                                ship_size = None
                                honk_ongoing = False
                                start_honk_time = None
                                start_blast_time = None
                                blast_ongoing = False
                                blast_leeway = BASE_BLAST_LEEWAY

                            else:
                                start_blast_time = None
                                blast_ongoing = False
                                blast_leeway = BASE_BLAST_LEEWAY

                # blast already ended and recorded, wait until honk is over or new one found
                elif blasts_in_honk:
                    inbetween_time = now - last_blast_end

                    # honk is over, post it if it's valid
                    if inbetween_time > MAX_BLAST_GAP:

                        # quickly check if we still are dealing with a special case
                        if HonkState.two_second_gap and len(blasts_in_honk) != 2:
                            HonkState.two_second_gap = False

                        # only care about length of first blast if only one recorded
                        if len(blasts_in_honk) == 1:
                            honk_time = round(first_blast_time, 2)
                        else:
                            honk_time = round(now - start_honk_time, 2)

                        self.poster.socials_queue.put(
                            (
                                ship_size,
                                start_honk_time,
                                honk_time,
                                blasts_in_honk,
                                self.decipherer,
                            )
                        )

                        # and reset variables for next loop
                        ship_size = None  # should work without it, but good to be safe
                        honk_ongoing = False
                        start_honk_time = None
                        start_blast_time = None
                        last_blast_end = None
                        blast_ongoing = False
                        blasts_in_honk = []
                        two_prolonged_timer = None
                        first_blast_time = None

    def find_horns(self):
        """Code to find honks in different frequency bands"""
        # get the latest chunk sized clip of audio and the data from it
        audio = self.stream.read(CHUNK)
        data = numpy.frombuffer(audio, dtype=numpy.int16).astype(numpy.float32)

        # normalise the data to between -1 to 1
        data /= 32768.0

        mags = numpy.abs(scipy.fft.rfft(data))

        # get our sub-sets of small, medium and large boat horn ranges
        small_mags = mags[self.small_horn_bins]
        medium_mags = mags[self.medium_horn_bins]
        large_mags = mags[self.large_horn_bins]

        # number of samples to keep from each sub-set
        small_to_keep = int(numpy.round(len(small_mags) * SAMPLE_PERCENT))
        medium_to_keep = int(numpy.round(len(medium_mags) * SAMPLE_PERCENT))
        large_to_keep = int(numpy.round(len(large_mags) * SAMPLE_PERCENT))

        # get the top x% from each sub-set, to ensure strong data and reduce bias
        small_top = numpy.sort(small_mags)[::-1][:small_to_keep]
        medium_top = numpy.sort(medium_mags)[::-1][:medium_to_keep]
        large_top = numpy.sort(large_mags)[::-1][:large_to_keep]

        # get averages for each boat size
        # see NOTE: x average and NOTE: percent average in #12
        small_boat_energy = numpy.mean(small_top)
        medium_boat_energy = numpy.mean(medium_top)
        large_boat_energy = numpy.mean(large_top)

        total_energy = numpy.mean(mags[self.all_bins])

        # compare boat energy to overall
        small_energy = small_boat_energy / total_energy
        medium_energy = medium_boat_energy / total_energy
        large_energy = large_boat_energy / total_energy

        # make dictionary of energies (can be expanded easily)
        horns = {"small": small_energy, "medium": medium_energy, "large": large_energy}

        # get our highest
        # NOTE: tie prioritises smaller boat, but an exact tie is unlikely to happen
        loudest_label = max(horns, key=horns.get)
        loudest_value = horns[loudest_label]

        # return label if over minimum
        if loudest_value > MIN_MAGS:
            return loudest_label
        # or empty string if one isn't found
        return ""

    def shutdown(self):
        """for safe shutdown when keyboard interrupt"""
        self.poster.shutdown()


class HonkDecipherer:
    """Class to decipher honks into their messages"""

    # see NOTE: unknown signal in #27
    DEFAULT_MESSAGE = "unknown signal"

    EMPTY = "empty"

    def decipher_blasts(self, blasts_in_honk):
        """
        Takes the blasts found and returns what the message from them is.
        (Presently ignores bad weather signals if there is overlap)
        (Could also have message with all possibilities?)
        """
        if not blasts_in_honk:
            return self.EMPTY

        current_blast = blasts_in_honk[0]

        next_blasts = None

        # if only one left
        if len(blasts_in_honk) != 1:
            next_blasts = blasts_in_honk[1:]

        # start
        if current_blast == "short":
            if next_blasts is None:
                return "Altering course to starboard."
            else:
                return self.decipher_short_blasts(next_blasts, 1)

        elif current_blast == "long":
            if next_blasts is None:
                # weather exception here, the only COLREGs exception that I haven't covered
                if not HonkState.heard_blind:  # first time hearing it
                    HonkState.heard_blind = True
                    return "Is anyone around this blind corner?"
                else:  # heard it before, must be response
                    HonkState.heard_blind = False
                    return "I am around this blind corner!"
            else:
                return self.decipher_long_blasts(next_blasts, 11)

        elif current_blast == "departure":
            return "I am a Cook Strait ferry leaving the harbour!"

        else:  # distress
            return "THIS IS A GENERAL DISTRESS SIGNAL!"

    def decipher_short_blasts(self, blasts_in_honk, state):
        """Decipher honks that start with a short"""

        # should always be at least one blast to look at when method is called
        # if not, it is not a known signal
        if not blasts_in_honk:
            return self.DEFAULT_MESSAGE

        current_blast = blasts_in_honk[0]

        next_blasts = None

        # if more than one left, get rest to pass forward
        if len(blasts_in_honk) != 1:
            next_blasts = blasts_in_honk[1:]

        # short ...
        if state == 1:
            if current_blast == "short":
                if next_blasts is None:
                    return "Altering course to port."
                else:
                    return self.decipher_short_blasts(next_blasts, 2)
            else:
                return self.decipher_short_blasts(next_blasts, 3)

        # short, short ...
        elif state == 2:
            if current_blast == "short":
                if next_blasts is None:
                    return "Operating astern propulsion."
                else:
                    return self.decipher_short_blasts(next_blasts, 4)
            else:
                if next_blasts is None:
                    return "I need your attention!"

        # short, long ...
        elif state == 3 and current_blast == "short" and next_blasts is None:
            return "I'm not moving in this bad weather with my anchor down."

        # short, short, short ...
        elif state == 4:
            if current_blast == "short":
                if next_blasts is None:
                    return "I am a pilot vessel performing pilotage duties."
                else:
                    return self.decipher_short_blasts(next_blasts, 5)
            else:
                return self.decipher_sos_blasts(next_blasts, 6)

        # short, short, short, short ...
        elif state == 5 and current_blast == "short":
            return "What are your intentions, other boat?"

        # broke the chain, don't know what it is
        return self.DEFAULT_MESSAGE

    def decipher_sos_blasts(self, blasts_in_honk, state):
        """
        The blasts seem to be leading to an SOS signal, check it
        Already found "short, short, short, long"
        """

        # unknown signal
        if not blasts_in_honk:
            return self.DEFAULT_MESSAGE

        current_blast = blasts_in_honk[0]

        next_blasts = None

        # if more than one left, get rest to pass forward
        if len(blasts_in_honk) != 1:
            next_blasts = blasts_in_honk[1:]

        # short, short, short, long ...
        if state == 6 and current_blast == "long" and next_blasts is not None:
            # could be SOS
            return self.decipher_sos_blasts(next_blasts, 7)

        # short, short, short, long, long ...
        elif state == 7 and current_blast == "long" and next_blasts is not None:
            return self.decipher_sos_blasts(next_blasts, 8)

        # short, short, short, long, long, long ...
        elif state == 8 and current_blast == "short" and next_blasts is not None:
            return self.decipher_sos_blasts(next_blasts, 9)

        # short, short, short, long, long, long, short ...
        elif state == 9 and current_blast == "short" and next_blasts is not None:
            return self.decipher_sos_blasts(next_blasts, 10)

        # short, short, short, long, long, long, short, short ...
        elif state == 10 and current_blast == "short" and next_blasts is None:
            return "SOS!"

        # broke the chain, don't know what it is
        return self.DEFAULT_MESSAGE

    def decipher_long_blasts(self, blasts_in_honk, state):
        """Decipher honks that start with a long"""

        # unknown signal
        if not blasts_in_honk:
            return self.DEFAULT_MESSAGE

        current_blast = blasts_in_honk[0]

        next_blasts = None
        # if more than one left, get rest to pass forward
        if len(blasts_in_honk) != 1:
            next_blasts = blasts_in_honk[1:]

        # long ...
        if state == 11:
            if current_blast == "short":
                return self.decipher_long_blasts(next_blasts, 12)
            else:
                if next_blasts is None and HonkState.two_second_gap:
                    HonkState.two_second_gap = False  # reset
                    return "I am sitting still in bad weather."
                elif not HonkState.two_second_gap:  # only triggers if above not called
                    return (
                        "Two boats are approaching a blind corner and "
                        "are signaling to each other that they're there."
                    )
                else:
                    return self.decipher_long_blasts(next_blasts, 15)

        # long, short ...
        elif state == 12:
            if current_blast == "short":
                if next_blasts is None:
                    return "It's bad weather and I'm special (see notes)."
                else:
                    return self.decipher_long_blasts(next_blasts, 13)
            else:
                return self.decipher_long_blasts(next_blasts, 14)

        # long, short, short ...
        elif state == 13 and current_blast == "short" and next_blasts is None:
            return "I am being towed in bad weather."

        # long, short, long ...
        elif (
            state == 14
            and current_blast == "short"
            and next_blasts is None
            and HonkState.heard_overtake
        ):
            HonkState.heard_overtake = False
            return "Acknowledged, you can overtake me."  # dependent on overtaking signal first

        # long, long ...
        elif state == 15 and current_blast == "short":
            if next_blasts is None:
                HonkState.heard_overtake = True
                return "I want to overtake another boat on their starboard."
            if len(next_blasts) == 4:  # if length of possible response left
                # check if it really is the exact correct response
                if (
                    next_blasts[0] == "long"
                    and next_blasts[1] == "short"
                    and next_blasts[2] == "long"
                    and next_blasts[3] == "short"
                ):
                    return (
                        "One boat wants to overtake another on its starboard; "
                        "they acknowledged and confirmed."
                    )
            else:
                return self.decipher_long_blasts(next_blasts, 16)

        # long, long, short ...
        elif state == 16 and current_blast == "short":
            if next_blasts is None:
                HonkState.heard_overtake = True
                return "I want to overtake another boat on their port."
            elif len(next_blasts) == 4:  # if length of possible response left
                # check if it really is the exact correct response
                if (
                    next_blasts[0] == "long"
                    and next_blasts[1] == "short"
                    and next_blasts[2] == "long"
                    and next_blasts[3] == "short"
                ):
                    return (
                        "One boat wants to overtake another on its port; "
                        "they acknowledged and confirmed."
                    )

        # broke the chain, don't know what it is
        return self.DEFAULT_MESSAGE


class HonkPoster:
    """Class to post about honks onto social media"""

    def __init__(self):
        """Initialise the class"""

        # load env variables
        load_dotenv()

        # initialize a thread-safe background queue to avoid stream overflows when posting
        self.socials_queue = queue.Queue()

        # start background worker thread immediately
        self.worker_thread = threading.Thread(target=self._socials_worker, daemon=True)
        self.worker_thread.start()

        # load Bluesky, so we can post onto it
        # give it a request timeout so it can't get stuck on a post forever
        self.bluesky_client = Client(request=Request(timeout=Timeout(10.0)))
        self.bluesky_name = os.getenv("BLUESKY_NAME")
        self.bluesky_pass = os.getenv("BLUESKY_PASS")
        self.bluesky_profile = self.bluesky_client.login(self.bluesky_name, self.bluesky_pass)

    def _socials_worker(self):
        """Runs continuously in the background, executing slow network operations."""
        while True:
            # Pulls text out of the queue. If empty, waits.
            post_data = self.socials_queue.get()
            if post_data is None:
                return  # shutdown signal to be safe

            # takes the tuple data and unpacks it
            ship_size, start_honk_time, honk_time, blasts_in_honk, decipherer = post_data

            try:
                self.post_to_socials(
                    ship_size, start_honk_time, honk_time, blasts_in_honk, decipherer
                )

            # I know, catching broad exceptions isn't great,
            # but I didn't have time to figure out exact errors before hand-in
            except Exception as e:
            	print(f"[Poster Error] Could not post to Bluesky: {e}")
            finally:
                self.socials_queue.task_done()

    def shutdown(self):
        """For safe shutdown after keyboard interrupt"""
        self.socials_queue.put(None)
        self.worker_thread.join()

    def post_to_socials(self, ship_size, start_honk_time, honk_time, blasts_in_honk, decipherer):
        """Code for social media posting"""
        # figure out what the blasts meant
        signal = decipherer.decipher_blasts(blasts_in_honk)

        # see NOTE: unknown signal in #27
        # only want to post if we found a real signal
        if signal not in (HonkDecipherer.DEFAULT_MESSAGE, HonkDecipherer.EMPTY):
            # make time into formats we want
            start_time_human = datetime.fromtimestamp(start_honk_time).strftime(
                "%d-%m-%Y %H:%M:%S"
            )
            self.post_to_bluesky(
                ship_size, start_time_human, honk_time, len(blasts_in_honk), signal
            )
            # more social medias could be added here

    def post_to_bluesky(self, ship_size, start_time_human, honk_time, blast_count, signal):
        """Code to post the information to Bluesky"""
        bluesky_text = ""

        # add honks
        for _ in range(blast_count):
            bluesky_text += "HONK "
        bluesky_text = bluesky_text[:-1]  # remove last space
        bluesky_text += "!\n"

        # now info on ship
        bluesky_text += "At " + start_time_human + " Wellington time, "
        bluesky_text += "a " + ship_size + " boat honked!\n"

        # info on honk
        bluesky_text += "This honk took " + str(honk_time) + " seconds. "
        bluesky_text += "It meant: " + signal

        # post it!
        self.bluesky_client.send_post(text=bluesky_text)


# start the program's main if this is the first thing you called
if __name__ == "__main__":
    detect = HonkDetector()
    try:
        detect.detector_main()
    # specifically KeyboardInterrupt because the program is made to run while True on command line
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        detect.shutdown()
