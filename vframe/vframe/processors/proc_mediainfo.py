def mediainfo(fp_in, raw=False):
  """Get media info using pymediainfo"""
  
  media_info_raw = MediaInfo.parse(fp_in).to_data()
  media_info = {}

  if raw:
    for d in media_info_raw['tracks']:
      if d['track_type'] == 'Video':
        media_info['video'] = d
      elif d['track_type'] == 'Audio':
        media_info['audio'] = d
  else:
    for d in media_info_raw['tracks']:
      if d['track_type'] == 'Video':
        media_info['video'] = {
          'codec_cc': d['codec_cc'],
          'duration': int(d['duration']),
          'display_aspect_ratio': float(d['display_aspect_ratio']),
          'width': int(d['width']),
          'height': int(d['height']),
          'frame_rate': float(d['frame_rate']),
          'frame_count': int(d['frame_count']),
          }
  
  return media_info


  """
  Mediainfo:
  codec_cc
  display_aspect_ratio
  frame_count
  width
  height
  frame_rate
  duration (in ms)
  """

  """
  format_url
  proportion_of_this_stream
  frame_count
  stream_identifier
  other_scan_type
  count_of_stream_of_this_kind
  interlacement
  codec_settings__cabac
  codec_id_info
  chroma_subsampling
  other_maximum_bit_rate
  other_kind_of_stream
  codec_cc
  track_type
  count
  codec_settings
  encoded_date
  format_settings__cabac
  other_bit_depth
  stored_height
  other_format_settings__reframes
  bits__pixel_frame
  format_profile
  other_stream_size
  other_track_id
  resolution
  format
  color_space
  sampled_height
  other_display_aspect_ratio
  other_width
  rotation
  codec_family
  framerate_mode_original
  other_interlacement
  other_height
  codec
  display_aspect_ratio
  duration
  bit_rate
  frame_rate_mode
  height
  sampled_width
  maximum_bit_rate
  pixel_aspect_ratio
  codec_id
  scan_type
  codec_url
  codec_info
  other_duration
  codec_settings_refframes
  streamorder
  tagged_date
  track_id
  other_format_settings__cabac
  format_settings__reframes
  other_codec
  bit_depth
  format_info
  other_frame_rate
  commercial_name
  frame_rate
  stream_size
  colorimetry
  other_frame_rate_mode
  internet_media_type
  format_settings
  kind_of_stream
  codec_id_url
  other_resolution
  codec_profile
  width
  other_bit_rate
  mediainfo
  None
  """

  