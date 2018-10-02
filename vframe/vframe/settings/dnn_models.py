"""Data container models"""


class MediaHashMap:
  """Data container for SHA256 encoded media items"""

  def __init__(self, sha256, ext):
    self.sha256 = sha256
    self.sha256_tree = fiox.sha256_tree(sha256)
    self.ext = ext
  
  def filepath(self, dir_source):
    return join(dir_source, sha256_tree, '{}.{}'.format(sha256, ext))




# add this network to network list
# conver to YAML or build big model
class NetModelOpts:
	
	name = 'DNN Model'

	def __init__(self, name):
		self.name = name



class CaffeNetOpts(NetModelOpts):

	# Caffe DNN models require prototxt, caffemodel, and txt file of classes

	def __init__(self, fp_prototxt, fp_model, fp_classes, mean=None, size=None):
		super().__init()
		self.fp_prototxt = fp_prototxt
		self.fp_model = fp_model
		self.fp_classes = classes
		self.mean = mean
		self.size = size



class Item(object):

    """
    Items are the things that pass through a chair pipeline.

    They can contain an image, some data, and multiple drawing backends supporting multiple layers.

    They require a click context, and either a cv2input or a size (w,h).

    Other arguments are optional. All items are guaranteed to have a name; one
    will be created if none is provided.
    """

    def __init__(self, ctx, cv2input=None, cv2input_gray=None, data=None, parent=None, size=None, name=None, no_draw=False, cv_features2d=None):
        self.ctx = ctx
        self.parent = parent
        self.data = data.copy() if data else {}
        self._cv2input_gray = cv2input_gray

        if cv2input is not None:
            cv2input = cv2input.copy()
            cv2input.flags['WRITEABLE'] = 0

        self.cv2input = cv2input

        # DrawBackend objects, keyed by class name
        self._draw_backends = {}

        # DrawBackend argument dictionaries, keyed by class name
        self._draw_backends_args = {}

        # DrawMetaLayer objects, keyed by layer name
        self._draw_layers = OrderedDict()

        # keypoints and descriptors, keyed by data_key
        self.cv_features2d = cv_features2d if cv_features2d is not None else {}

        # descriptor matches, keyed by data_key
        self.cv_descriptor_matches = {}

        # set of related items (used for SVG transclusions)
        self.related_items = set()

        # dictionary of lists of regions, by name
        self.regions = OrderedDict()

        # None or tuple of (x, y, scale, no_radii)
        self._draw_relative = None

        # None or tuple of (item, (x, y, scale, no_radii))
        self._proxy_draw = None

        # None or tuple of (item, (x, y), (xscale, yscale))
        self._proxy_regions = None

        if cv2input is not None:
            assert size is None, "size and cv2input are mutually exclusive"
            if len(cv2input.shape) > 2:
                self.height, self.width, self.channels = cv2input.shape
            else:
                (self.height, self.width), self.channels = cv2input.shape, 1
        else:
            self.width, self.height = size
            self.channels = '?'

        # ensure we don't clobber ancestors' metadata
        metadata = self.metadata.copy()

        # override metadata template with this item's own values
        metadata.update(
            width    = self.width,
            height   = self.height,
            channels = self.channels,
            ctime    = time.time(),
            )

        if 'gp_ctime' not in metadata:
            metadata['gp_ctime'] = metadata['ctime']

        self.data['metadata'] = metadata

        if name is not None:
            self.name = name
            self.data['metadata']['name'] = name

        elif 'name' in self.data['metadata']:
            self.name = self.data['metadata']['name']

        else:
            self.name = "Untitled %s" % (UNTITLED.next(),)

        if not no_draw:
            if parent:
                self.add_draw_backends_from_other( parent )
            else:
                self.add_draw_backend( DrawBackend_cv2, fmt='tiff', cv2output=None, gray=False )

    def __repr__(self):
        return "<{type} {name!r} {width}x{height} backends={backends!r} labels={labels!r} at 0x{id:x}>".format(
                    type     = type(self).__name__,
                    width    = self.width,
                    height   = self.height,
                    name     = self.name,
                    id       = id(self),
                    backends = self._draw_backends.values(),
                    labels   = self.data.get('labels', []),
                    )

    @property
    def labels(self):
        return self.data.setdefault('labels',[])

    @property
    def draw(self):
        return self._draw_backends or self._proxy_draw

    @property
    def size(self):
        return (self.width, self.height)

    @property
    def aspect_ratio(self):
        return float(max(self.size)) / min(self.size)

    @property
    def size_precise(self):
        if hasattr(self,'_size_precise'):
            return self._size_precise
        else:
            return (self.width, self.height)

    @property
    def metadata(self):
        return self.data.setdefault('metadata', {})

    @property
    def age(self):
        """
        Seconds since the item was created
        """
        ctime = self.metadata.get('gp_ctime', 0)
        if ctime:
            return time.time() - ctime
        else:
            return 0

    @property
    def fps(self):
        """
        Returns 1/self.grandparent.age (potentially the fps)
        """
        return 1 / self.grandparent.age

    @property
    def pdb(self):
        """
        call pdb.set_trace
        """
        import pdb
        pdb.set_trace()

    @property
    def cv2input_gray(self):
        if self._cv2input_gray is None and self.cv2input is not None:
            self._cv2input_gray = cv2.cvtColor(self.cv2input, cv2.COLOR_BGR2GRAY)
            self._cv2input_gray.flags['WRITEABLE'] = 0
        return None if self._cv2input_gray is None else self._cv2input_gray.copy()

    @property
    def cv2output(self):
        draw_backend_cv2 = self.get_draw_backend( DrawBackend_cv2 )
        if draw_backend_cv2:
            return draw_backend_cv2.layer.img
        else:
            return None
#            click.echo("""
#            warning: Something is accessing item.cv2output while the cv2
#            drawing backend was not loaded.
#
#            Without the cv2 drawing backend, each user of cv2output receives
#            its own new copy of the source image, which is probably not what
#            you want.
#
#            To load the cv2 drawing backend, add draw_cv2 early in the pipeline
#            (usually immediately after the open command).""")
#
#            raise Exception("hopefully we've removed all the cases where this can happen easily... comment this out if necessary")
#
#            return self.cv2input.copy()

    @property
    def orig_full_path(self):
        return self.data.get('metadata',{}).get('orig_full_path')

    @property
    def orig_full_path_split(self):
        return filter(None, self.orig_full_path.split('/'))

    @property
    def orig_bitmap(self):
        """
        Return (filename, bytes) of the original file, or synthesize something
        like it if the item did not originate from an image file on disk.
        """
        orig_path = self.orig_full_path
        cv2input  = self.cv2input

        if (orig_path # we have an orig_path
            and not self.metadata.get('is_video_frame') # and we aren't a video frame
            and self.size == self.grandparent.size): # and we haven't been resized
                # then we return the contents of the original file
                with open(orig_path) as fh:
                    return (os.path.basename(orig_path), fh.read())
        elif cv2input is not None:
            # otherwise, we must encode our cv2input
            ret, data_np = cv2.imencode('.tiff', self.cv2input)
            assert ret, "cv2.imencode failed"
            data_str = ''.join(map(chr, data_np.reshape(-1)))
            return ("%s.tiff" % (self.name,), data_str)
        else:
            return None, None

    @property
    def orig_bitmap_data_uri(self):
        """
        Return data: URI of the original file, or synthesize something
        like it if the item did not originate from an image file on disk.
        """
        name, data = self.orig_bitmap
        ext = name.split('.')[-1].lower()
        assert ext in ('jpg', 'jpeg', 'tif', 'tiff', 'png'), "unsupported extension in %r" % (name,)
        if ext == 'jpg':
            ext = 'jpeg'
        elif ext == 'tif':
            ext = 'tiff'
        return 'data:image/%s;base64,%s' % ( ext, data.encode('base64') )

    def add_data(self, key, data):
        if key not in self.data:
            self.data[key] = []

        self.data[key].append( data )

    def add_label(self, name, value=None):
        self.add_data('labels', (name, value))

    def get_label(self, name, default=None):
        "Returns first label with a given name, or default value"
        return next((v for k,v in self.data.get('labels',[]) if k == name), default)

    def get_draw_backend(self, backend):
        return self._draw_backends.get( backend.__name__ if backend is not None else None )
    
    def add_draw_backend(self, backend, **kwargs):
        assert backend.__name__ not in self._draw_backends, "attempted to add backend %r a second time" % (backend.__name__,)
        if self.regions:
            # this warning is here because this creates a confusing situation
            # where regions have a different backend than their parent item.
            # fixme: should we have an assert instead of a warning? or should we add the backend to the regions too?
            print("warning: adding draw backend {} after regions already exist".format(backend.__name__,))
        self._draw_backends_args[backend.__name__] = kwargs
        self._draw_backends[backend.__name__] = backend(self, **kwargs)

    def remove_drawing_backends(self):
        self._draw_backends.clear()
        self._draw_backends_args.clear()

    def add_draw_backends_from_other(self, other, **overrides):
        """
        This copies drawing backends and their arguments from another item, and
        creates matching ones on this item.

        It optionally allows some arguments to be replaced, if those arguments
        exist already on the other item.

        The override feature exists so that commands like contact_sheet can use
        the first item as a reference item, while changing some of its options
        (eg, import_cv2_item).
        """
        for backend in other._draw_backends.values():
            backend_cls  = backend.__class__
            backend_args = other._draw_backends_args[ backend_cls.__name__ ].copy()
            for k, v in overrides.items():
                if k in backend_args:
                    backend_args[k] = v
            self.add_draw_backend( backend_cls, **backend_args )

    def as_json(self, pretty=True):
        try:
            return json.dumps(self.data,
                              indent = 2 if pretty else None,
                              sort_keys = True,
                              default = lambda a: a.tolist(), # for numpy arrays
                              )
        except TypeError(ex)    :
            click.echo("%r" % (ex,))
            if self.ctx.obj['verbose'] >= 3:
                click.echo(repr(self.data))
            else:
                click.echo("To see the full object that is not serializing, run chair with -vvv")
            sys.exit(1)

    @property
    def pretty_json(self):
        return self.as_json(pretty=True)

    def get_layer(self, name):
        """
        Get a drawing layer
        """
        # FIXME: maybe this should make a new DrawMetaLayer every time, so that
        # when a layer has already been drawn to before a new backend was added
        # subsequent uses of that layer will be able to draw on the new backend
        # too?
        # that would require the backends to handle layer reuse themselves, though.

        if self._proxy_draw:
            other_item, rel_draw_args = self._proxy_draw
            assert other_item.draw, "can't proxy draw to an item that has no drawing backends"
            res = other_item.get_layer(name).draw_relative(*rel_draw_args)
        else:
            assert self.draw, "attempted to get a drawing layer, but there is no drawing backend"
            if name not in self._draw_layers:
                self._draw_layers[name] = DrawMetaLayer(name, self._draw_backends.values())
            res = self._draw_layers[name]
        if self._draw_relative:
            res = res.draw_relative(*self._draw_relative)
        return res

    def get_layer_on_item(self, name):
        """
        Same as get_layer on items; this is overridden on Region objects.
        """
        return self.get_layer(name)

    def save_orig(self, get_path):
        # FIXME: make this call on grandparent, and memoize it or something?
        # many items might share a parent!

        orig_full_path = self.grandparent.orig_full_path

        if orig_full_path and not self.metadata.get('is_video_frame'):
            new_orig_path = get_path(ext=None, name=orig_full_path, prefix='originals')
            shutil.copyfile( orig_full_path, new_orig_path )
        else:
            cv2input = self.cv2input
            if cv2input is not None:
                cv2.imwrite( get_path(ext='tiff', prefix='originals'), cv2input )


    def save_image(self, get_path):
        for backend in self._draw_backends.values():
            backend.save_image(get_path)

    def display(self, name):
        for backend in self._draw_backends.values():
            backend.display(name)

    def display_done(self):
        for backend in self._draw_backends.values():
            backend.display_done()

    @property
    def grandparent(self):
        """
        This is currently only used by save_orig, and this is the only user of
        parent besides __init__'s call to add_draw_backends_from_other.

        FIXME: perhaps parent should be replaced with draw_like for the latter
        purpose, and then save_orig could rely on an explicit orig_full_path
        argument to __init__, and then this grandparent business could go away.
        """
        return self.parent.grandparent if self.parent else self

    def make_output_item(self, copy_features=False):
        """
        Create a new item using the current item's output as the new item's
        input.
        """
        data = self.data.copy()
        if self.orig_full_path:
            del data['metadata']['orig_full_path']

        cv2data = self.cv2output

        if cv2data is None:
            PIL_backend = self._draw_backends.get('DrawBackend_PIL')
            cairo_backend = self._draw_backends.get('DrawBackend_Cairo')
            if PIL_backend:
                cv2data = PIL_backend.as_cv2
            elif cairo_backend:
                cv2data = cairo_backend.as_cv2
            else:
                raise Exception("can't create output item without cv2 or PIL backend loaded")

        new_item = Item( self.ctx, cv2input=cv2data, data=data, name=self.name, parent=self )
        if copy_features:
            new_item.cv_features2d.update(self.cv_features2d)
            new_item.cv_descriptor_matches.update(self.cv_descriptor_matches)
        return new_item

    def copy(self, cv2input=None, forget_parent=False, no_draw=False):

        if cv2input is None:
            cv2input = self.cv2input

        if cv2input is None:
            return Item( self.ctx, size=(self.width, self.height), data=self.data, parent=None if forget_parent else self, no_draw=no_draw )
        else:
            return Item( self.ctx, cv2input=cv2input, data=self.data, parent=None if forget_parent else self, no_draw=no_draw )

    def resize_canvas(self, w, h, background, placement):

        assert placement in "NW N NE E SE S SW W center".split()

        if 'W' in placement:
            X = 0
        elif 'E' in placement:
            X = w - self.width
        else:
            X = (w / 2) - (self.width / 2)

        if 'N' in placement:
            Y = 0
        elif 'S' in placement:
            Y = h - self.height
        else:
            Y = (h / 2) - (self.height / 2)

        w,h = map(int_r,(w,h))
        new_img = np.zeros( (h, w, 3), np.uint8 )
        new_img[:] = background.bgr
        np_compose(new_img, self.cv2input, map(int_r,(Y,X)))
        new_item = self.copy(cv2input=new_img)
        draw = new_item.get_layer("pre-resize-canvas")
        draw.item_output((X,Y), self.size, self) # FIXME: this codepath is too slow
        return new_item

    def resize(self, w, h, proxy_draw=False, no_radii=False, proxy_regions=False):

        # because ffmpeg doesn't like odd-sized frames, and cv2 wants ints
#        w += w % 2
#        h += h % 2
        # FIXME the above causes problems with contactsheet at least
        w, h = int(w), int(h)
        wF = w / float(self.width)
        hF = h / float(self.height)
        if 0 in (w,h):
            print("resize: ignoring zero value (attempted to resize to {}x{}".format(w,h))
            return self
        try:
            cv2img = cv2.resize(self.cv2input, (w, h))
        except Exception, ex:
            print("resize: %r (attempted to resize to %sx%s)" % (ex, w, h))
            return self
        new_item = Item(self.ctx, cv2img, name=self.name, data=self.data, parent=self, no_draw=True )
        if not proxy_draw:
            new_item.add_draw_backends_from_other( self, cv2output=None, bg_color=None )
        else:
            new_item._proxy_draw = (self, (0, 0, (1.0 / wF), no_radii))

        if proxy_regions:
            new_item._proxy_regions = (self, (0,0), (1.0/wF, 1.0/hF))

        new_item.import_labeled_rois( self.export_labeled_rois, wF, hF )

        return new_item

    def define_region(self, name, *xywh, **kw):

        x, y, w, h = map(int, xywh)

        if kw.get('crop'):
            if x < 0:
                w += x
                x = 0
            if x + w > self.width:
                w = self.width - x
            if y < 0:
                h += y
                y = 0
            if y + h > self.height:
                h = self.height - y

        if self._proxy_regions:
            other, (xoffset, yoffset), (xscale, yscale) = self._proxy_regions
            return other.define_region(name, x*xscale+xoffset, y*yscale+yoffset, w*xscale, h*yscale)

        assert x + w <= self.width and y + h <= self.height, (x,y,w,h,self.width,self.height)

        region_list = self.regions.setdefault(name, [])
        number      = len(region_list) + 1
        title       = "%s %s region %s" % (self.name, name, number)
        region      = Region( self, title, x, y, w, h )
        self.data.setdefault('region_defs',{}).setdefault(name,[]).append( (x, y, w, h) )
        region_list.append( region )
        return region

    @property
    def export_labeled_rois(self):
        res = {}
        for name, regions in self.regions.items():
            res[name] = []
            for region in regions:
                x, y = map(float,region.region_pos)
                w, h = map(int,  region.size_precise)
                labels = region.data.get('labels',[])
                res[name].append( ((x,y,w,h), labels) )
        return res
    
    def import_labeled_rois(self, data, x_scale=1, y_scale=1):
        for name, regions in data.items():
            for (x,y,w,h), labels in regions:
                new_region = self.define_region(name, x*x_scale, y*y_scale, w*x_scale, h*y_scale)
                for label in labels:
                    new_region.add_label(*label)

    def to_dict(self):
        """
        Returns a pickleable dict which can be used to reinstantiate an Item
        like this one. Currently only used by parallelize command.
        """
        return dict(
            name          = self.name,
            cv2input      = self.cv2input,
            data          = self.data,
            cv_features2d = self.cv_features2d,
            )

class Region(Item):

    def __init__(self, item, name, x, y, w, h):
        self.region_pos = (x, y)
        self._size_precise = (w, h)
        x, y, w, h = map(int, (x, y, w, h))
        self.draw_on_parent = False
        self.proxy_labels = False
        cv2input  = item.cv2input
        cv2input  = cv2input[y:y+h, x:x+w] if cv2input is not None else None
        cv2output = item.cv2output
        cv2output = cv2output[y:y+h, x:x+w] if cv2output is not None else None
        cv2input_gray = item._cv2input_gray
        cv2input_gray = cv2input_gray[y:y+h, x:x+w] if cv2input_gray is not None else None
        size = {} if cv2input is not None else dict(size=(w,h))
        super(Region, self).__init__(item.ctx, cv2input, name=name, parent=item, no_draw=True, **size)
        self.add_draw_backends_from_other( item, cv2output=cv2output, cv2input_gray=cv2input_gray, no_bg=True, bg=None )

    @property
    def draw(self):
        return self._draw_backends or self._proxy_draw or self.parent.draw

    def get_layer(self, name):
        if self.draw_on_parent:
            return self.get_layer_on_item(name)
        else:
            return super(Region, self).get_layer(name)

    def add_label(self, name, value=None):
        if self.proxy_labels:
            self.parent.add_label(name, value)
        else:
            super(Region, self).add_label(name, value)

    def get_layer_on_item(self, name):
        """
        Return a relative drawing proxy for this region on the grandparent canvas.

        This is useful for drawing in a region but not cropping to it.
        """
        item = self
        offsets = []
        while isinstance(item, Region):
            offsets.append(item.region_pos)
            item = item.parent
        draw = item.get_layer(name)
        for offset in reversed(offsets):
            # fixme: should really compute a single offset, to not have nested proxies
            draw = draw.draw_relative(*offset)
        if self._draw_relative:
            draw = draw.draw_relative(*self._draw_relative)
        return draw

    @property
    def quadrant(self):
        """
        Determine which quadrant of the parent object this region falls within,
        if any. Quadrants are numbered 0 through 3, clockwise from the top left.

        If the region is not entirely contained within one quadrant, return -1.

        This function should really have tests! It has been tested manually
        using this pipeline, however:

        chair open /HTSLAM/input/Videos/To-Process/Everyday_Life_test_HD.mov downsize -w 320 resend 4 contact_sheet -w 320 -h 180 -c 2 -r 2 dest2src cv_haar { roi draw_border -C red label -v 'Q{item.quadrant}' quadrant label -v '{item.region_pos}' pos draw_labels --bg black -n quadrant -n pos --anonymous } watch { roi --one-per-quadrant draw_border -C lime  } watch
        """

        x, y = self.region_pos
        w, h = self.size
        W, H = self.parent.size

        if 0 <= x < W and \
           0 <= y < H and \
           w < (W/2 - (x % (W/2) ) ) and \
           h < (H/2 - (y % (H/2) ) ):
            if y < H/2:
                if x < W/2:
                    return 0
                else:
                    return 1
            else:
                if x >= W/2:
                    return 2
                else:
                    return 3
        return -1

    def enlarge(self, left, top, right, bottom):
        """
        On the first non-region ancestor of this region, define and return a
        region that is like this region but enlarged by some amount.
        """
        item = self
        ancestor_positions = []

        while isinstance(item, Region):
            ancestor_positions.append(item.region_pos)
            item = item.parent

        X, Y = map(sum, zip(*ancestor_positions))
        x = max(0, X - left)
        y = max(0, Y - top)
        w = min(item.width-x,  self.width  + X - x + right )
        h = min(item.height-y, self.height + Y - y + bottom)
        return item.define_region("%s_enlarged_%s_%s_%s_%s" % (self.name, left, top, right, bottom), x, y, w, h)

	