"""Object that passes through the chair/click pipeline"""

class ChairItem(object):

	def __init__(self, ctx, media_record):
		"""Items that pass through the chair pipeline"""
		self._ctx = ctx
		self._media_record = media_record
		self._sha256 = self._media_record.sha256

	@property
	def ctx(self):
		return self._ctx

	@property
	def item(self):
		return self._media_record

	@property
	def record(self):
		"""both refer to same data"""
		return self._media_record

	@property
	def media_record(self):
		"""both refer to same data"""
		return self._media_record

	@property
	def sha256(self):
		return self._sha256

	@property
	def verified(self):
		return self._media_record.verified

	@property
	def format(self):
		return self._media_record.media_format

	@property
	def media_format(self):
		"""alternate name for same data"""
		return self._media_record.media_format
	