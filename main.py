import praw
import pandas as pd
from IPython.display import display
from gtts import gTTS
import os
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, ColorClip, concatenate_videoclips
import audioread
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import time
from faster_whisper import WhisperModel
import json
import numpy as np
from pydub import AudioSegment
import random

#setting imagemagick binary path since I am on windows
from moviepy.config import change_settings
#if on windows set this to your image magick abs path
#change_settings({"IMAGEMAGICK_BINARY": r""})

#setting wkdir
os.chdir('set working directory')

#setting up whisper model for later use
model_size = 'medium'
model = WhisperModel(model_size)

#putting in information for reddit API
reddit = praw.Reddit(
    client_id="Client ID",
    client_secret="Client Secret",
    user_agent="user_agent",
    username = 'reddit username',
    password = 'reddit password'
)

#setting up dictionary for DF set up
dict = {'id' : [],
        'title': [],
        'likes': []}

#appending top 10 liked posts right now to dictionary
for submission in reddit.subreddit("AmItheAsshole").hot(limit=10):
    dict['id'].append(submission.id)
    dict['title'].append(submission.title)
    dict['likes'].append(submission.score)


#putting dictionary into a dataframe for ease of viewing and sorting it by top likes then getting the top liked post
posts = pd.DataFrame(dict)
posts = posts.sort_values(by = ['likes'], ascending=False)
topPost = posts.iloc[0, posts.columns.get_loc('id')]
display(posts)

readFile = open('required/last.txt', 'r')
editFile = open('required/last.txt', 'r+')
tempVar = readFile.read()
print(topPost)
if topPost == tempVar:
    print('ID of this post and last are the same')
    readFile.close()
else:
    
  #setting description of tts, title then description of the post
  description = reddit.submission(id = topPost).title + ' ' +  reddit.submission(id = topPost).selftext + " Who's in the wrong? Comment your answer!"
  descriptionText = open('items/' + reddit.submission(id = topPost).id + '.txt', 'x')
  descriptionText.write(description)
  descriptionText.close()

  #putting it there so that I can look at the post on reddit
  print(reddit.submission(id = topPost).url)

  #language for gtts
  language = 'en'

  #Using gtts to get mp3 of the reddit post
  ttsVid = gTTS(text = description, lang = language, slow = False)

  #name of the gtts mp3 file and saving it 
  ttsName = 'items/AITA '+ reddit.submission(id = topPost).id
  fttsName = 'Finished/AITA '+ reddit.submission(id = topPost).id
  ttsVid.save(ttsName + '.mp3')

  #Speeding up the audio for the mp3 file since normal speed is a little too slow and getting background music added
  musicNumber = random.randint(1,4)
  background = AudioSegment.from_file(f'required/backgroundmusic{musicNumber}.mp3', format = 'mp3')
  speedAud = AudioSegment.from_file(ttsName + '.mp3', format = 'mp3')
  finalSpeed = speedAud.speedup(playback_speed=1.25)
  overlay = finalSpeed.overlay(background, position = 0, loop = 1)
  overlay.export(ttsName + ".wav", format="wav")

  print('Finished TTS Audio')

  #cutting base video to the length of the gtts file
  cutVid = 'items/AITA' + reddit.submission(id = topPost).id + ' cut.mp4'
  audDur = audioread.audio_open(ttsName + '.wav').duration
  whatVid = input('Minecraft or Satisfying?\n')
  minecraft = VideoFileClip('required/minecraft.mp4')
  minecraftAud = int(minecraft.duration - audDur)
  minecraftDur = random.randint(0,minecraftAud)
  minecraftFinal = int(audDur + minecraftDur)
  print(minecraftDur)
  satisfy = VideoFileClip('required/satisfy.mp4')
  satisfyAud = int(satisfy.duration - audDur)
  satisfyDur = random.randint(0,satisfyAud)
  satisfyFinal = int(audDur + satisfyDur)
  print(satisfyDur)

  if whatVid == 'Minecraft' or whatVid == 'minecraft':
     ffmpeg_extract_subclip('required/minecraft.mp4', minecraftDur, minecraftFinal, targetname=cutVid)
  else :
     ffmpeg_extract_subclip('required/satisfy.mp4', satisfyDur, satisfyFinal, targetname=cutVid)

  #setting up files for whisper to transcribe the audio
  vid = VideoFileClip(cutVid)
  aud = AudioFileClip(ttsName + '.wav')

  #transcribing the audio using whisper and dumping the timestamp level words into a json
  segments, info = model.transcribe(ttsName + '.wav', word_timestamps=True)
  segments = list(segments)
  wordlevel_info = []

  for segment in segments:
      for word in segment.words:
        wordlevel_info.append({'word':word.word,'start':word.start,'end':word.end})

  with open('items/data.json', 'w') as f:
     json.dump(wordlevel_info, f, indent=4)

  #making the words into lines from the json file above and using max char limit to decide how many words are on the screen at a time
  def split_text_into_lines(data):

      MaxChars = 5
      #maxduration in seconds
      MaxDuration = 2.5
      #Split if nothing is spoken (gap) for these many seconds
      MaxGap = 1.5

      subtitles = []
      line = []
      line_duration = 0
      line_chars = 0


      for idx,word_data in enumerate(data):
          word = word_data["word"]
          start = word_data["start"]
          end = word_data["end"]

          line.append(word_data)
          line_duration += end - start

          temp = " ".join(item["word"] for item in line)


          # Check if adding a new word exceeds the maximum character count or duration
          new_line_chars = len(temp)

          duration_exceeded = line_duration > MaxDuration
          chars_exceeded = new_line_chars > MaxChars
          if idx>0:
            gap = word_data['start'] - data[idx-1]['end']
            maxgap_exceeded = gap > MaxGap
          else:
            maxgap_exceeded = False


          if duration_exceeded or chars_exceeded or maxgap_exceeded:
              if line:
                  subtitle_line = {
                      "word": " ".join(item["word"] for item in line),
                      "start": line[0]["start"],
                      "end": line[-1]["end"],
                      "textcontents": line
                  }
                  subtitles.append(subtitle_line)
                  line = []
                  line_duration = 0
                  line_chars = 0


      if line:
          subtitle_line = {
              "word": " ".join(item["word"] for item in line),
              "start": line[0]["start"],
              "end": line[-1]["end"],
              "textcontents": line
          }
          subtitles.append(subtitle_line)

      return subtitles

  print('Json Made')

  #getting the json information and splitting the words into the respective lines with timestamps for the subtitle creation
  linelevel_subtitles = split_text_into_lines(wordlevel_info)

  for line in linelevel_subtitles:
     json_str = json.dumps(line, indent=4)

  #setting parameters of the text to be generated on the screen
  def create_caption(textJSON, framesize,font = "Arial",color='white', highlight_color='yellow',stroke_color='black',stroke_width=1.5):
      wordcount = len(textJSON['textcontents'])
      full_duration = textJSON['end']-textJSON['start']

      word_clips = []
      xy_textclips_positions =[]

      x_pos = 0
      y_pos = 0
      line_width = 0  # Total width of words in the current line
      frame_width = framesize[0]
      frame_height = framesize[1]

      x_buffer = frame_width*1/10

      max_line_width = frame_width - 2 * (x_buffer)

      fontsize = int(frame_height * 0.075) #7.5 percent of video height

      space_width = ""
      space_height = ""

      for index,wordJSON in enumerate(textJSON['textcontents']):
        duration = wordJSON['end']-wordJSON['start']
        word_clip = TextClip(wordJSON['word'], font = font,fontsize=fontsize, color=color,stroke_color=stroke_color,stroke_width=stroke_width).set_start(textJSON['start']).set_duration(full_duration)
        word_clip_space = TextClip(" ", font = font,fontsize=fontsize, color=color).set_start(textJSON['start']).set_duration(full_duration)
        word_width, word_height = word_clip.size
        space_width,space_height = word_clip_space.size
        if line_width + word_width+ space_width <= max_line_width:
              # Store info of each word_clip created
              xy_textclips_positions.append({
                  "x_pos":x_pos,
                  "y_pos": y_pos,
                  "width" : word_width,
                  "height" : word_height,
                  "word": wordJSON['word'],
                  "start": wordJSON['start'],
                  "end": wordJSON['end'],
                  "duration": duration
              })

              word_clip = word_clip.set_position((x_pos, y_pos))
              word_clip_space = word_clip_space.set_position((x_pos+ word_width, y_pos))

              x_pos = x_pos + word_width+ space_width
              line_width = line_width+ word_width + space_width
        else:
              # Move to the next line
              x_pos = 0
              y_pos = y_pos+ word_height+10
              line_width = word_width + space_width

              # Store info of each word_clip created
              xy_textclips_positions.append({
                  "x_pos":x_pos,
                  "y_pos": y_pos,
                  "width" : word_width,
                  "height" : word_height,
                  "word": wordJSON['word'],
                  "start": wordJSON['start'],
                  "end": wordJSON['end'],
                  "duration": duration
              })

              word_clip = word_clip.set_position((x_pos, y_pos))
              word_clip_space = word_clip_space.set_position((x_pos+ word_width , y_pos))
              x_pos = word_width + space_width


        word_clips.append(word_clip)
        word_clips.append(word_clip_space)


      for highlight_word in xy_textclips_positions:

        word_clip_highlight = TextClip(highlight_word['word'], font = font,fontsize=fontsize, color=highlight_color,stroke_color=stroke_color,stroke_width=stroke_width).set_start(highlight_word['start']).set_duration(highlight_word['duration'])
        word_clip_highlight = word_clip_highlight.set_position((highlight_word['x_pos'], highlight_word['y_pos']))
        word_clips.append(word_clip_highlight)

      return word_clips,xy_textclips_positions

  print('Line level words done')

  #getting the cut video and the subtitle video ready for the final part where the audio, subtitles, and the background video get spliced together
  input_video = VideoFileClip(cutVid)
  frame_size = input_video.size

  all_linelevel_splits = []

  print('Starting Captions')

  for line in linelevel_subtitles:
    out_clips,positions = create_caption(line,frame_size)

    max_width = 0
    max_height = 0

    for position in positions:

      x_pos, y_pos = position['x_pos'],position['y_pos']
      width, height = position['width'],position['height']

      max_width = max(max_width, x_pos + width)
      max_height = max(max_height, y_pos + height)

    color_clip = ColorClip(size=(int(max_width*1.1), int(max_height*1.1)),
                         color=(64, 64, 64))
    color_clip = color_clip.set_opacity(0)
    color_clip = color_clip.set_start(line['start']).set_duration(line['end']-line['start'])

    clip_to_overlay = CompositeVideoClip([color_clip] + out_clips)
    clip_to_overlay = clip_to_overlay.set_position("center")


    all_linelevel_splits.append(clip_to_overlay)

  print('Captions Made')

  input_video_duration = input_video.duration

  print('Starting Final Video Compilation')

  final_video = CompositeVideoClip([input_video] + all_linelevel_splits)

  final_video = final_video.set_audio(aud)

  final_video.write_videofile(fttsName + " final.mp4", threads = 24, codec = 'libx264', fps = 30)

  print('Video Finished')

  print('Starting Youtube')

  temp = VideoFileClip(fttsName + ' final.mp4')
  postId = reddit.submission(id = topPost).id

  if temp.duration/2 < 60:
      x = temp.duration/2
      print(x)
      ffmpeg_extract_subclip(fttsName + ' final.mp4', 0, x, targetname= 'parts/AITA ' + postId + ' part1.mp4')
      ffmpeg_extract_subclip(fttsName + ' final.mp4', x, temp.duration, targetname= 'parts/AITA ' + postId + ' part2.mp4')
  elif temp.duration/3 < 60:
      x = temp.duration/3
      x1 = x + x
      print(x)
      ffmpeg_extract_subclip(fttsName + ' final.mp4', 0, x, targetname= 'parts/AITA ' + postId + ' part1.mp4')
      ffmpeg_extract_subclip(fttsName + ' final.mp4', x, x1, targetname= 'parts/AITA ' + postId + ' part2.mp4')
      ffmpeg_extract_subclip(fttsName + ' final.mp4', x1, temp.duration, targetname= 'parts/AITA ' + postId + ' part3.mp4')
  elif temp.duration/4 < 60:
      x = temp.duration/4
      x1 = x + x
      x2 = x + x + x
      print(x)
      ffmpeg_extract_subclip(fttsName + ' final.mp4', 0, x, targetname= 'parts/AITA ' + postId + ' part1.mp4')
      ffmpeg_extract_subclip(fttsName + ' final.mp4', x, x1, targetname= 'parts/AITA ' + postId + ' part2.mp4')
      ffmpeg_extract_subclip(fttsName + ' final.mp4', x1, x2, targetname= 'parts/AITA ' + postId + ' part3.mp4')
      ffmpeg_extract_subclip(fttsName + ' final.mp4', x2, temp.duration, targetname= 'parts/AITA ' + postId + ' part4.mp4')
  elif temp.duration/5 < 60:
      x = temp.duration/4
      x1 = x + x
      x2 = x + x + x
      x3 = x + x + x + x
      print(x)
      ffmpeg_extract_subclip(fttsName + ' final.mp4', 0, x, targetname= 'parts/AITA ' + postId + ' part1.mp4')
      ffmpeg_extract_subclip(fttsName + ' final.mp4', x, x1, targetname= 'parts/AITA ' + postId + ' part2.mp4')
      ffmpeg_extract_subclip(fttsName + ' final.mp4', x1, x2, targetname= 'parts/AITA ' + postId + ' part3.mp4')
      ffmpeg_extract_subclip(fttsName + ' final.mp4', x2, x3, targetname= 'parts/AITA ' + postId + ' part4.mp4')
      ffmpeg_extract_subclip(fttsName + ' final.mp4', x3, temp.duration, targetname= 'parts/AITA ' + postId + ' part5.mp4')
  else:
      print('add /6 to the abhorant ifelse statement')

  print('Youtube Done')

  print(f'All Done, the id of the video is {postId}')

  #time.sleep(15)

  #os.remove(cutVid)
  #os.remove(ttsName + '.wav')

  print('Yay')
  editFile.truncate(0)
  readFile.close()
  editFile.write(topPost)
  editFile.close()