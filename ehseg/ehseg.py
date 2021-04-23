import os
import sys
import shutil
import rasterio
import cv2 as cv
import tempfile
import datetime as dt
import numpy as np
from skimage.filters import sobel, scharr, \
    roberts, prewitt, laplace, gaussian, \
    threshold_local
from skimage.morphology import skeletonize, \
    thin, medial_axis
from sklearn.preprocessing import MinMaxScaler


def cal_emb_egm(img_array):
    """A function to calculate ensemble (mean) of
    different edge gradient magnitude
    The methods: sobel, scharr, roberts,
    prewitt, laplace, and laplace gaussian
    Args:
        img_array (numpy.ndarray): the image array
    Returns:
        numpy.ndarray: the mean of edge gradient magnitudes
    """
    img_sobel = sobel(img_array)
    img_scharr = scharr(img_array)
    img_roberts = roberts(img_array)
    img_prewitt = prewitt(img_array)
    img_laplac = laplace(img_array)
    img_laplac_gauss = laplace(gaussian(img_array))
    img_egm_mean = np.mean([img_sobel, img_scharr,
                            img_roberts, img_prewitt,
                            img_laplac, img_laplac_gauss], axis=0)
    # img_egm_max = np.maximum.reduce([img_sobel, img_scharr,
    #                                  img_roberts, img_prewitt,
    #                                  img_laplac, img_laplac_gauss])
    return img_egm_mean


def cal_accum_egm(img_array,
                  n_iter=3,
                  max_window=11):
    """A function to calculate edge gradient magnitude
    Args:
        img_array (numpy.ndarray): the image array read by rasterio.
            This input should be [B, G, R, NIR].
        n_iter (int): the number of iteration for mean-shift.
        max_window (int): the maximum window size for mean-shift loop.
    Returns:
        numpy.ndarray, list (numpy.ndarray):
            the mean of edge gradient magnitudes and the filtered bands
    """
    # Get channels, cols, rows and bands
    [_, cols, rows] = img_array.shape
    b1, b2, b3, b4 = img_array

    # Calculate vegetation indices as added variables
    # NDVI and SAVI for commonly usage
    ndvi = (b4 - b3) / (b4 + b3)
    savi = ((b4 - b3) / (b4 + b3 + 1)) * 2

    # Step 1
    # Use mean-shift algorithm with n iterations to smooth image and filter out noise
    for i in range(n_iter):
        # Scale to int8 for opencv processing
        scaler = MinMaxScaler(feature_range=(0, 255))
        # Get min and max
        b1_norm = scaler.fit_transform(b1.astype(np.float64).reshape(cols * rows, 1)).astype(np.uint8)
        b2_norm = scaler.fit_transform(b2.astype(np.float64).reshape(cols * rows, 1)).astype(np.uint8)
        b3_norm = scaler.fit_transform(b3.astype(np.float64).reshape(cols * rows, 1)).astype(np.uint8)
        b4_norm = scaler.fit_transform(b4.astype(np.float64).reshape(cols * rows, 1)).astype(np.uint8)
        ndvi_norm = scaler.fit_transform(ndvi.astype(np.float64).reshape(cols * rows, 1)).astype(np.uint8)
        savi_norm = scaler.fit_transform(savi.astype(np.float64).reshape(cols * rows, 1)).astype(np.uint8)

        # keep records for min and max for original bands
        b1_min, b1_max = np.min(b1), np.max(b1)
        b2_min, b2_max = np.min(b2), np.max(b2)
        b3_min, b3_max = np.min(b3), np.max(b3)
        b4_min, b4_max = np.min(b4), np.max(b4)
        ndvi_min, ndvi_max = np.min(ndvi), np.max(ndvi)
        savi_min, savi_max = np.min(savi), np.max(savi)

        rgb_norm = cv.merge([b1_norm.reshape(cols, rows),
                             b2_norm.reshape(cols, rows),
                             b3_norm.reshape(cols, rows)])

        veg_norm = cv.merge([b4_norm.reshape(cols, rows),
                             ndvi_norm.reshape(cols, rows),
                             savi_norm.reshape(cols, rows)])

        # Apply Mean-shift filter in two groups of bands
        # rgb_ms = cv.pyrMeanShiftFiltering(rgb_norm, max_window - i, max_window - i,
        #                                   termcrit=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
        #                                             1, max_window - i))
        # veg_ms = cv.pyrMeanShiftFiltering(veg_norm, max_window - i, max_window - i,
        #                                   termcrit=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
        #                                             1, max_window - i))

        rgb_ms = cv.pyrMeanShiftFiltering(rgb_norm, max_window - i, max_window - i)
        veg_ms = cv.pyrMeanShiftFiltering(veg_norm, max_window - i, max_window - i)

        # Get each bands and convert to original range (i.e., original band data range)
        b1_ms, b2_ms, b3_ms = cv.split(rgb_ms)
        b4_ms, ndvi_ms, savi_ms = cv.split(veg_ms)
        b1 = MinMaxScaler(feature_range=(b1_min, b1_max)).\
            fit_transform(b1_ms.reshape(cols * rows, 1).astype(np.float32)).\
            reshape(cols, rows)
        b2 = MinMaxScaler(feature_range=(b2_min, b2_max)).\
            fit_transform(b2_ms.reshape(cols * rows, 1).astype(np.float32)).\
            reshape(cols, rows)
        b3 = MinMaxScaler(feature_range=(b3_min, b3_max)).\
            fit_transform(b3_ms.reshape(cols * rows, 1).astype(np.float32)).\
            reshape(cols, rows)
        b4 = MinMaxScaler(feature_range=(b4_min, b4_max)).\
            fit_transform(b4_ms.reshape(cols * rows, 1).astype(np.float32)).\
            reshape(cols, rows)
        ndvi = MinMaxScaler(feature_range=(ndvi_min, ndvi_max)).\
            fit_transform(ndvi_ms.reshape(cols * rows, 1).astype(np.float32)).\
            reshape(cols, rows)
        savi = MinMaxScaler(feature_range=(savi_min, savi_max)).\
            fit_transform(savi_ms.reshape(cols * rows, 1).astype(np.float32)).\
            reshape(cols, rows)

    # Step 2
    # EGM calculation using different edge detection convolution kernels
    # laplacian, laplacian of gaussian, freichen, sobel, prewitt, roberts
    b1_egm = cal_emb_egm(b1)
    b2_egm = cal_emb_egm(b2)
    b3_egm = cal_emb_egm(b3)
    b4_egm = cal_emb_egm(b4)
    ndvi_egm = cal_emb_egm(ndvi)
    savi_egm = cal_emb_egm(savi)
    egm_mean = np.mean([b1_egm, b2_egm, b3_egm,
                        b4_egm, ndvi_egm, savi_egm], axis=0)
    return egm_mean, [b1, b2, b3, b4, ndvi, savi]


def edge_highlight(img_array,
                   n_iter=3, max_window=11,
                   window_size_thred=101):
    """A function to highlight the edges of an image
    Args:
        img_array (numpy.ndarray): the image array read by rasterio.
            This input should be [B, G, R, NIR].
        n_iter (int): the iteration number for cal_accum_egm.
        max_window (int): the maximum window size for cal_accum_egm.
        window_size_thred (int): the size of window to get adaptive threshold.
    Returns:
        numpy.ndarray: the edge highlighted images
    """
    start_dt = dt.datetime.now()
    print("Single edge highlight start: {}".format(start_dt))
    egm, img_ms = cal_accum_egm(img_array, n_iter=n_iter,
                                max_window=max_window)
    threds = threshold_local(egm, window_size_thred)
    skl_adp = skeletonize(egm > threds, method='lee')
    skl_adp = thin(skl_adp)
    skl_adp = medial_axis(skl_adp)
    print("Single edge highlight end: {}".
          format(dt.datetime.now() - start_dt))
    return np.transpose(np.dstack(img_ms), (2, 0, 1)) * (1 - skl_adp)


def edge_highlight_2s(img1_array,
                      img2_array,
                      n_iter=3, max_window=11,
                      window_size_thred=101,
                      method='mean'):
    """A function to highlight the edges based on double images.
    The images are usually the combination of growing season and off
    season image.
    Args:
        img1_array (numpy.ndarray): the image array read by rasterio.
            This input should be [B, G, R, NIR].
        img2_array (numpy.ndarray): the image array read by rasterio.
            This input should be [B, G, R, NIR].
        n_iter (int): the iteration number for cal_accum_egm.
        max_window (int): the maximum window size for cal_accum_egm.
        window_size_thred (int): the size of window to get adaptive threshold.
        method (str); method to get edges. ['mean', 'separate']
            "mean" to take the mean of gs and os;
            "separate" to get edges from gs and os separately;
    Returns:
        numpy.ndarray, numpy.ndarray: two edge-highlighted image stacks
    """
    if method == 'mean':
        # Calculate the skeleton of the objects.
        start_dt = dt.datetime.now()
        print("Mean edge highlight start: {}".format(start_dt))
        egm_os, img1_ms = cal_accum_egm(img1_array, n_iter=n_iter,
                                        max_window=max_window)
        egm_gs, img2_ms = cal_accum_egm(img2_array, n_iter=n_iter,
                                        max_window=max_window)
        egm_s_mean = np.mean([egm_os, egm_gs], axis=0)
        threds = threshold_local(egm_s_mean, window_size_thred)
        skl_adp = skeletonize(egm_s_mean > threds, method='lee')
        skl_adp = thin(skl_adp)
        skl_adp = medial_axis(skl_adp)
        print("Mean edge highlight end: {}".
              format(dt.datetime.now() - start_dt))
    elif method == 'separate':
        start_dt = dt.datetime.now()
        print("Separate edge highlight start: {}".format(start_dt))
        egm_os, img1_ms = cal_accum_egm(img1_array, n_iter=n_iter,
                                        max_window=max_window)
        threds = threshold_local(egm_os, window_size_thred)
        skl_adp_os = skeletonize(egm_os > threds, method='lee')
        egm_gs, img2_ms = cal_accum_egm(img2_array, n_iter=n_iter,
                                        max_window=max_window)
        threds = threshold_local(egm_gs, window_size_thred)
        skl_adp_gs = skeletonize(egm_gs > threds, method='lee')
        skl_adp = skl_adp_os + skl_adp_gs
        skl_adp = thin(skl_adp)
        skl_adp = medial_axis(skl_adp)
        print("Separate edge highlight end: {}".
              format(dt.datetime.now() - start_dt))
    else:
        print('No such option, just mean or separate.')

    return np.transpose(np.dstack(img1_ms), (2, 0, 1)) * (1 - skl_adp), \
           np.transpose(np.dstack(img2_ms), (2, 0, 1)) * (1 - skl_adp)


def ehseg(img_paths,
          dst_path,
          opt_name='segments',
          bands=[1, 2, 3, 4],
          grassbin='/Applications/GRASS-7.9.app/Contents/MacOS/Grass.sh',  # for Mac
          gisbase='/Applications/GRASS-7.9.app/Contents/Resources',  # for Mac
          n_iter=3, max_window=11,
          window_size_thred=101,
          method='mean',
          ram=8,
          threshold=0.2,
          similarity="manhattan",
          minsize=10,
          iterations=20,
          vectorize=True,
          simplify_thred=0.0,
          keep=False):
    """
    Args:
        img_paths (list or str): be a list of image paths
            or a single image path.
            One for single image, two for double images.
        opt_name (str): output name for segments.
        bands (list of int): the band index to read. Should be [B, G, R, NIR].
        grassbin (str): the path of GRASS installation.
        gisbase (str): the gisbase path of GRASS GIS.
        dst_path (str): the destination path for outputs.
        n_iter (int): the iteration number for edge_highlight or edge_highlight_2s.
        max_window (int): the maximum window size for edge_highlight or edge_highlight_2s.
        window_size_thred (int): the size of window to get adaptive threshold.
            This is for edge_highlight or edge_highlight_2s.
        method (str): method to get edges for edge_highlight_2s. ['mean', 'separate']
            "mean" to take the mean of gs and os;
            "separate" to get edges from gs and os separately;
        ram (int): the number of RAM in G to use.
        threshold (float): a threshold for segmentation, more details refers to i.segment GRASS manual.
        similarity (str): the method to calculate similarity between segments. euclidean or manhattan.
        minsize (int): the minimum size of segments.
        iterations (int): number of iterations for segmentation algorithm to converge.
        vectorize (bool): the option to vectorize the segments or not.
        simplify_thred (float): the threshold of Douglas-Peucker Algorithm to simplify the vectorized segments.
            If the segments will be used as interim output (e.g. get the edge of fields),
            it is not a good idea to simplify them.
        keep (bool): the option to keep the intermediate results.
    """
    # Link GRASS GIS
    os.environ['GISBASE'] = gisbase
    os.environ['GRASSBIN'] = grassbin
    gpydir = os.path.join(gisbase, "etc", "python")
    sys.path.append(gpydir)
    from grass_session import Session
    import grass.script as gscript

    if isinstance(img_paths, str):
        print('Use single image.')
        # Read images
        with rasterio.open(img_paths, "r") as src:
            img_array = src.read(bands)
            meta = src.meta
        # Calculate skeleton
        img_hlt = edge_highlight(img_array, n_iter=n_iter,
                                 max_window=max_window,
                                 window_size_thred=window_size_thred)

        # Save out skl for GRASS GIS
        meta.update({'count': 6, 'dtype': 'float64'})
        with rasterio.open(os.path.join(dst_path, 'img_hlt.tif'), 'w', **meta) as dst:
            dst.write(img_hlt)

        # Call GRASS GIS to do segmentation
        with Session(gisdb=dst_path,
                     location="ehseg",
                     mapset='PERMANENT',
                     create_opts="EPSG:{}".format(meta['crs'].to_epsg())):
            gscript.run_command('r.in.gdal',
                                input=os.path.join(dst_path, 'img_hlt.tif'),
                                output='img_hlt',
                                overwrite=True)
            gscript.run_command('g.region', raster='img_hlt.1')
            gscript.run_command('i.group',
                                group='group',
                                input='img_hlt.1,img_hlt.2,img_hlt.3,'
                                      'img_hlt.4,img_hlt.5,img_hlt.6,')
            gscript.run_command('i.segment',
                                threshold=threshold,
                                method="region_growing",  # mean_shift
                                similarity=similarity,
                                minsize=minsize,
                                memory=1024 * ram,
                                iterations=iterations,
                                group="group",
                                output='segments',
                                goodness='goodness',
                                overwrite=True)
            gscript.run_command('r.out.gdal',
                                flags='m',
                                input='segments',
                                output=os.path.join(dst_path, '{}.tif'.format(opt_name)),
                                overwrite=True)
            if vectorize:
                gscript.run_command('r.to.vect',
                                    input='segments',
                                    output='segments',
                                    type='area',
                                    overwrite=True)
                # simplify the polygons
                if simplify_thred > 0.0:
                    gscript.run_command('v.generalize',
                                        input='segments',
                                        output='segments_dg',
                                        method='douglas',
                                        threshold=5,
                                        overwrite=True)
                    gscript.run_command('v.out.ogr',
                                        input='segments_dg',
                                        format='GeoJSON',
                                        output=os.path.join(dst_path, '{}.geojson'.format(opt_name)),
                                        overwrite=True)
                else:
                    gscript.run_command('v.out.ogr',
                                        input='segments',
                                        format='GeoJSON',
                                        output=os.path.join(dst_path, '{}.geojson'.format(opt_name)),
                                        overwrite=True)

    elif isinstance(img_paths, list) and \
            all(isinstance(elem, str) for elem in img_paths) and \
            len(img_paths) == 2:
        print('Use double image.')
        # Read images
        with rasterio.open(img_paths[0], "r") as src:
            img1_array = src.read(bands)
            meta = src.meta
        with rasterio.open(img_paths[1], "r") as src:
            img2_array = src.read(bands)
        # Calculate skeleton
        img1_hlt, img2_hlt = edge_highlight_2s(img1_array, img2_array,
                                               n_iter=n_iter,
                                               max_window=max_window,
                                               window_size_thred=window_size_thred,
                                               method=method)

        # Save out skl for GRASS GIS
        meta.update({'count': 6, 'dtype': 'float64'})
        with rasterio.open(os.path.join(dst_path, 'img1_hlt.tif'), 'w', **meta) as dst:
            dst.write(img1_hlt)
        with rasterio.open(os.path.join(dst_path, 'img2_hlt.tif'), 'w', **meta) as dst:
            dst.write(img2_hlt)

        with Session(gisdb=dst_path,
                     location="ehseg",
                     mapset='PERMANENT',
                     create_opts="EPSG:{}".format(meta['crs'].to_epsg())):
            gscript.run_command('r.in.gdal',
                                input=os.path.join(dst_path, 'img1_hlt.tif'),
                                output='img1_hlt',
                                overwrite=True)
            gscript.run_command('r.in.gdal',
                                input=os.path.join(dst_path, 'img2_hlt.tif'),
                                output='img2_hlt',
                                overwrite=True)
            gscript.run_command('g.region', raster='img1_hlt.1')
            gscript.run_command('i.group',
                                group='group',
                                input='img1_hlt.1,img1_hlt.2,img1_hlt.3,'
                                      'img1_hlt.4,img1_hlt.5,img1_hlt.6,'
                                      'img2_hlt.1,img2_hlt.2,img2_hlt.3,'
                                      'img2_hlt.4,img2_hlt.5,img2_hlt.6')
            gscript.run_command('i.segment',
                                threshold=threshold,
                                method="region_growing",  # mean_shift
                                similarity=similarity,
                                minsize=minsize,
                                memory=1024 * ram,
                                iterations=iterations,
                                group="group",
                                output='segments',
                                goodness='goodness',
                                overwrite=True)
            gscript.run_command('r.out.gdal',
                                flags='m',
                                input='segments',
                                output=os.path.join(dst_path, '{}.tif'.format(opt_name)),
                                overwrite=True)
            if vectorize:
                gscript.run_command('r.to.vect',
                                    flags='s',
                                    input='segments',
                                    output='segments',
                                    type='area',
                                    overwrite=True)
                # simplify the polygons
                if simplify_thred > 0.0:
                    gscript.run_command('v.generalize',
                                        input='segments',
                                        output='segments_dg',
                                        method='douglas',
                                        threshold=5,
                                        overwrite=True)
                    gscript.run_command('v.out.ogr',
                                        input='segments_dg',
                                        format='GeoJSON',
                                        output=os.path.join(dst_path, '{}.geojson'.format(opt_name)),
                                        overwrite=True)
                else:
                    gscript.run_command('v.out.ogr',
                                        input='segments',
                                        format='GeoJSON',
                                        output=os.path.join(dst_path, '{}.geojson'.format(opt_name)),
                                        overwrite=True)
    else:
        print("Not valid image paths.")
        sys.exit(-1)

    # Clean the intermediate results
    if not keep:
        # edge highlight images
        if os.path.isfile(os.path.join(dst_path, 'img_hlt.tif')):
            os.remove(os.path.join(dst_path, 'img_hlt.tif'))
        if os.path.isfile(os.path.join(dst_path, 'img1_hlt.tif')):
            os.remove(os.path.join(dst_path, 'img1_hlt.tif'))
        if os.path.isfile(os.path.join(dst_path, 'img2_hlt.tif')):
            os.remove(os.path.join(dst_path, 'img2_hlt.tif'))
        # GRASS GIS
        try:
            shutil.rmtree(os.path.join(dst_path, 'ehseg'))
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


def revalue_segments(segments_path,
                     categorical_mask_path,
                     dst_epsg,
                     category_filter=None,
                     thred_clean=500,
                     grassbin='/Applications/GRASS-7.8.app/Contents/MacOS/Grass.sh',  # for Mac
                     gisbase='/Applications/GRASS-7.8.app/Contents/Resources',  # for Mac
                     keep=False):
    """
    Args:
        segments_path (str): path of segments. Should be a vector.
        categorical_mask_path (str): path of mask image.
        category_filter (None or list): the category values for filtering.
        dst_epsg (int): the destination CRS in EPSG code.
            E.g. 4326 for Geographic coordinate system.
        thred_clean (int): threshold value in dst_epsg unit to clean the results.
        grassbin (str): the path of GRASS installation.
        gisbase (str): the gisbase path of GRASS GIS.
        keep (bool): the option to keep the original segments or not.
    Return:
        Save out file. Within the attribute table, b_value is the resigned value for each segment.
    """
    # Link GRASS GIS
    os.environ['GISBASE'] = gisbase
    os.environ['GRASSBIN'] = grassbin
    gpydir = os.path.join(gisbase, "etc", "python")
    sys.path.append(gpydir)
    from grass_session import Session
    import grass.script as gscript

    # Paths
    dst_dir = tempfile.TemporaryDirectory(dir='.')
    if keep:
        dst_path = segments_path.replace('segments', 'segments_revalue')
    else:
        dst_path = segments_path
    # Resign value of segments using GRASS GIS
    with Session(gisdb=dst_dir.name,
                 location="revalue_segments",
                 mapset='PERMANENT',
                 create_opts="EPSG:{}".format(dst_epsg)):
        gscript.run_command('v.import',
                            input=segments_path,
                            output='segments',
                            extent='input',
                            overwrite=True)
        gscript.run_command('r.import',
                            input=categorical_mask_path,
                            output='mask',
                            extent='input',
                            overwrite=True)
        gscript.run_command('g.region', raster='mask')
        gscript.run_command('r.null',
                            map='mask',
                            setnull=','.join([str(i) for i in category_filter]))
        gscript.run_command('r.to.vect',
                            flags='s',
                            input='mask',
                            output='mask',
                            type='area',
                            overwrite=True)
        gscript.run_command('v.overlay',
                            ainput='segments',
                            binput='mask',
                            operator='and',
                            output='segments_resign')
        gscript.run_command('v.clean',
                            input='segments_resign',
                            output='segments',
                            error='segment_error',
                            tool='rmarea',
                            threshold=thred_clean,
                            overwrite=True)
        gscript.run_command('v.out.ogr',
                            input='segments',
                            format='GeoJSON',
                            output=dst_path,
                            overwrite=True)

    # Clean temporary directory
    dst_dir.cleanup()

