#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <limits.h>
#include <float.h>
#include <iostream>
//#include "opencv2/core/core.hpp" 
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/nonfree/features2d.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <opencv2\opencv.hpp>
//using namespace cv;


#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

/** Label for pixels with undefined gradient. */
#define NOTDEF -1024.0
/** PI */
#ifndef M_PI
#define M_PI   3.14159265358979323846
#endif /* !M_PI */

#define M_1_4_PI 0.785398163

#define M_3_4_PI 2.35619449

#define M_1_8_PI 0.392699081
#define M_3_8_PI 1.178097245
#define M_5_8_PI 1.963495408
#define M_7_8_PI 2.748893572
#define M_1_9_PI  0.34906585  //20°
#define M_1_10_PI 0.314159265358979323846   //18°
#define M_1_12_PI 0.261799387   //15°
#define M_1_15_PI 0.20943951    //12°
#define M_1_18_PI 0.174532925   //10°
/** 3/2 pi */
#define M_3_2_PI 4.71238898038
/** 2 pi */
#define M_2__PI  6.28318530718
/** Doubles relative error factor
 */
#define RELATIVE_ERROR_FACTOR 100.0

struct point2i //(or pixel).
{
	int x,y;
};

struct point2d
{
	double x,y;
};

struct point3d
{
	double x,y;
	double r;
};

struct point3i
{
	int x,y;
	int z;
};

struct point2d1i
{
	double x,y;
	int z;
};

struct point1d1i
{
	double data;
	int cnt;
};

/*----------------------------------------------------------------------------*/
/** Rectangle structure: line segment with width.
 */
struct rect
{
  double x1,y1,x2,y2;  /* first and second point of the line segment */
  double width;        /* rectangle width */
  double x,y;          /* center of the rectangle */
  double theta;        /* angle */
  double dx,dy;        /* (dx,dy) is vector oriented as the line segment,dx = cos(theta), dy = sin(theta) */
  int   polarity;     /* if the arc direction is the same as the edge direction, polarity = 1, else if opposite ,polarity = -1.*/
  double prec;         /* tolerance angle */
  double p;            /* probability of a point with angle within 'prec' */
};

typedef struct
{
  double vx[4];  /* rectangle's corner X coordinates in circular order */
  double vy[4];  /* rectangle's corner Y coordinates in circular order */
  double ys,ye;  /* start and end Y values of current 'column' */
  int x,y;       /* coordinates of currently explored pixel */
} rect_iter;

typedef struct image_double_s
{
  double * data;
  int xsize,ysize;
} * image_double;


//==================================================================================================
//=============================miscellaneous functions==============================================
inline double min(double v1,double v2)
{
	return (v1<v2?v1:v2);
}
inline double max(double v1,double v2)
{
	return (v1>v2?v1:v2);
}
/** Compare doubles by relative error.

    The resulting rounding error after floating point computations
    depend on the specific operations done. The same number computed by
    different algorithms could present different rounding errors. For a
    useful comparison, an estimation of the relative rounding error
    should be considered and compared to a factor times EPS. The factor
    should be related to the cumulated rounding error in the chain of
    computation. Here, as a simplification, a fixed factor is used.
 */
int double_equal(double a, double b)
{
  double abs_diff,aa,bb,abs_max;

  /* trivial case */
  if( a == b ) return TRUE;

  abs_diff = fabs(a-b);
  aa = fabs(a);
  bb = fabs(b);
  abs_max = aa > bb ? aa : bb;

  /* DBL_MIN is the smallest normalized number, thus, the smallest
     number whose relative error is bounded by DBL_EPSILON. For
     smaller numbers, the same quantization steps as for DBL_MIN
     are used. Then, for smaller numbers, a meaningful "relative"
     error should be computed by dividing the difference by DBL_MIN. */
  if( abs_max < DBL_MIN ) abs_max = DBL_MIN;

  /* equal if relative error <= factor x eps */
  return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON); //RELATIVE_ERROR_FACTOR=100.0,
}

/*----------------------------------------------------------------------------*/
/** Absolute value angle difference.
 */
//得到2个弧度制角度的夹角的绝对值
double angle_diff(double a, double b)
{
  a -= b;
  while( a <= -M_PI ) a += M_2__PI;
  while( a >   M_PI ) a -= M_2__PI;
  if( a < 0.0 ) a = -a;
  return a;
}
/*----------------------------------------------------------------------------*/
/** Signed angle difference.
 */
double angle_diff_signed(double a, double b)
{
  a -= b;
  while( a <= -M_PI ) a += M_2__PI;
  while( a >   M_PI ) a -= M_2__PI;
  return a;
}

/*----------------------------------------------------------------------------*/
/** Fatal error, print a message to standard-error output and exit.
 */
void error(char * msg)
{
  fprintf(stderr,"circleDetection Error: %s\n",msg);
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*/
/** Computes Euclidean distance between point (x1,y1) and point (x2,y2).
 */
double dist(double x1, double y1, double x2, double y2)
{
  return sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) );
}

//向量内积
double dotProduct(point2d vec1, point2d vec2)
{
	return (vec1.x*vec2.x+vec1.y*vec2.y);
}

/*----------------------------------------------------------------------------*/
/** Copy one rectangle structure to another.
 */
void rect_copy(struct rect * in, struct rect * out)//in is the src, out is the dst
{
  /* check parameters */
  if( in == NULL || out == NULL ) error("rect_copy: invalid 'in' or 'out'.");

  /* copy values */
  out->x1 = in->x1;
  out->y1 = in->y1;
  out->x2 = in->x2;
  out->y2 = in->y2;
  out->width = in->width;
  out->x = in->x;
  out->y = in->y;
  out->theta = in->theta;
  out->dx = in->dx;
  out->dy = in->dy;
  out->polarity = in->polarity;
  out->prec = in->prec;
  out->p = in->p;
}

/*----------------------------------------------------------------------------*/
/** Interpolate y value corresponding to 'x' value given, in
    the line 'x1,y1' to 'x2,y2'; if 'x1=x2' return the smaller
    of 'y1' and 'y2'.

    The following restrictions are required:
    - x1 <= x2
    - x1 <= x
    - x  <= x2
 */
double inter_low(double x, double x1, double y1, double x2, double y2)
{
  /* check parameters */
  if( x1 > x2 || x < x1 || x > x2 )
    error("inter_low: unsuitable input, 'x1>x2' or 'x<x1' or 'x>x2'.");

  /* interpolation */
  if( double_equal(x1,x2) && y1<y2 ) return y1;
  if( double_equal(x1,x2) && y1>y2 ) return y2;
  return y1 + (x-x1) * (y2-y1) / (x2-x1);
}

/*----------------------------------------------------------------------------*/
/** Interpolate y value corresponding to 'x' value given, in
    the line 'x1,y1' to 'x2,y2'; if 'x1=x2' return the larger
    of 'y1' and 'y2'.

    The following restrictions are required:
    - x1 <= x2
    - x1 <= x
    - x  <= x2
 */
double inter_hi(double x, double x1, double y1, double x2, double y2)
{
  /* check parameters */
  if( x1 > x2 || x < x1 || x > x2 )
    error("inter_hi: unsuitable input, 'x1>x2' or 'x<x1' or 'x>x2'.");

  /* interpolation */
  if( double_equal(x1,x2) && y1<y2 ) return y2;
  if( double_equal(x1,x2) && y1>y2 ) return y1;
  return y1 + (x-x1) * (y2-y1) / (x2-x1);
}

/*----------------------------------------------------------------------------*/
/** Free memory used by a rectangle iterator.
 */
void ri_del(rect_iter * iter)
{
  if( iter == NULL ) error("ri_del: NULL iterator.");
  free( (void *) iter );
}

/*----------------------------------------------------------------------------*/
/** Check if the iterator finished the full iteration.

    See details in \ref rect_iter
 */
int ri_end(rect_iter * i)
{
  /* check input */
  if( i == NULL ) error("ri_end: NULL iterator.");

  /* if the current x value is larger than the largest
     x value in the rectangle (vx[2]), we know the full
     exploration of the rectangle is finished. */
  return (double)(i->x) > i->vx[2];
}

/*----------------------------------------------------------------------------*/
/** Increment a rectangle iterator.

    See details in \ref rect_iter
 */
void ri_inc(rect_iter * i)
{
  /* check input */
  if( i == NULL ) error("ri_inc: NULL iterator.");

  /* if not at end of exploration,
     increase y value for next pixel in the 'column' */
  if( !ri_end(i) ) i->y++;

  /* if the end of the current 'column' is reached,
     and it is not the end of exploration,
     advance to the next 'column' */
  while( (double) (i->y) > i->ye && !ri_end(i) )
    {
      /* increase x, next 'column' */
      i->x++;

      /* if end of exploration, return */
      if( ri_end(i) ) return;

      /* update lower y limit (start) for the new 'column'.

         We need to interpolate the y value that corresponds to the
         lower side of the rectangle. The first thing is to decide if
         the corresponding side is

           vx[0],vy[0] to vx[3],vy[3] or
           vx[3],vy[3] to vx[2],vy[2]

         Then, the side is interpolated for the x value of the
         'column'. But, if the side is vertical (as it could happen if
         the rectangle is vertical and we are dealing with the first
         or last 'columns') then we pick the lower value of the side
         by using 'inter_low'.
       */
      if( (double) i->x < i->vx[3] )
        i->ys = inter_low((double)i->x,i->vx[0],i->vy[0],i->vx[3],i->vy[3]);
      else
        i->ys = inter_low((double)i->x,i->vx[3],i->vy[3],i->vx[2],i->vy[2]);

      /* update upper y limit (end) for the new 'column'.

         We need to interpolate the y value that corresponds to the
         upper side of the rectangle. The first thing is to decide if
         the corresponding side is

           vx[0],vy[0] to vx[1],vy[1] or
           vx[1],vy[1] to vx[2],vy[2]

         Then, the side is interpolated for the x value of the
         'column'. But, if the side is vertical (as it could happen if
         the rectangle is vertical and we are dealing with the first
         or last 'columns') then we pick the lower value of the side
         by using 'inter_low'.
       */
      if( (double)i->x < i->vx[1] )
        i->ye = inter_hi((double)i->x,i->vx[0],i->vy[0],i->vx[1],i->vy[1]);
      else
        i->ye = inter_hi((double)i->x,i->vx[1],i->vy[1],i->vx[2],i->vy[2]);

      /* new y */
      i->y = (int) ceil(i->ys);
    }
}

/*----------------------------------------------------------------------------*/
/** Create and initialize a rectangle iterator.

    See details in \ref rect_iter
 */
rect_iter * ri_ini(struct rect * r)
{
  double vx[4],vy[4];
  int n,offset;
  rect_iter * i;

  /* check parameters */
  if( r == NULL ) error("ri_ini: invalid rectangle.");

  /* get memory */
  i = (rect_iter *) malloc(sizeof(rect_iter));
  if( i == NULL ) error("ri_ini: Not enough memory.");

  /* build list of rectangle corners ordered
     in a circular way around the rectangle */
  //从线段的起点(x1,y1)处的一端开始按照逆时针重构出矩形的四个定点
  vx[0] = r->x1 - r->dy * r->width / 2.0;
  vy[0] = r->y1 + r->dx * r->width / 2.0;
  vx[1] = r->x2 - r->dy * r->width / 2.0;
  vy[1] = r->y2 + r->dx * r->width / 2.0;
  vx[2] = r->x2 + r->dy * r->width / 2.0;
  vy[2] = r->y2 - r->dx * r->width / 2.0;
  vx[3] = r->x1 + r->dy * r->width / 2.0;
  vy[3] = r->y1 - r->dx * r->width / 2.0;

  /* compute rotation of index of corners needed so that the first
     point has the smaller x.

     if one side is vertical, thus two corners have the same smaller x
     value, the one with the largest y value is selected as the first.
   */
  if( r->x1 < r->x2 && r->y1 <= r->y2 ) offset = 0;
  else if( r->x1 >= r->x2 && r->y1 < r->y2 ) offset = 1;
  else if( r->x1 > r->x2 && r->y1 >= r->y2 ) offset = 2;
  else offset = 3;

  /* apply rotation of index. */
  for(n=0; n<4; n++)
    {
      i->vx[n] = vx[(offset+n)%4];
      i->vy[n] = vy[(offset+n)%4];
    }

  /* Set an initial condition.

     The values are set to values that will cause 'ri_inc' (that will
     be called immediately) to initialize correctly the first 'column'
     and compute the limits 'ys' and 'ye'.

     'y' is set to the integer value of vy[0], the starting corner.

     'ys' and 'ye' are set to very small values, so 'ri_inc' will
     notice that it needs to start a new 'column'.

     The smallest integer coordinate inside of the rectangle is
     'ceil(vx[0])'. The current 'x' value is set to that value minus
     one, so 'ri_inc' (that will increase x by one) will advance to
     the first 'column'.
   */
  i->x = (int) ceil(i->vx[0]) - 1;
  i->y = (int) ceil(i->vy[0]);
  i->ys = i->ye = -DBL_MAX;

  /* advance to the first pixel */
  ri_inc(i);

  return i;
}


/*----------------------------------------------------------------------------*/
/** Free memory used in image_double 'i'.
 */
void free_image_double(image_double i)
{
  if( i == NULL || i->data == NULL )
    error("free_image_double: invalid input image.");
  free( (void *) i->data );
  free( (void *) i );
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'.
 */
image_double new_image_double(int xsize, int ysize)
{
  image_double image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 ) error("new_image_double: invalid image size.");

  /* get memory */
  image = (image_double) malloc( sizeof(struct image_double_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (double *) calloc( (size_t) (xsize*ysize), sizeof(double) );
  if( image->data == NULL ) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'
    with the data pointed by 'data'.
 */
image_double new_image_double_ptr( int xsize,
                                          int ysize, double * data )
{
  image_double image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 )
    error("new_image_double_ptr: invalid image size.");
  if( data == NULL ) error("new_image_double_ptr: NULL data pointer.");

  /* get memory */
  image = (image_double) malloc( sizeof(struct image_double_s) );
  if( image == NULL ) error("not enough memory.");

  /* set image */
  image->xsize = xsize;
  image->ysize = ysize;
  image->data = data;

  return image;
}

//=================================================================================================================
//===========================================LSD functions=========================================================
/** ln(10) */
#ifndef M_LN10
#define M_LN10 2.30258509299404568402    //ln10
#endif /* !M_LN10 */

/** Label for pixels not used in yet. */
#define NOTUSED 0

/** Label for pixels already used in detection. */
#define USED    1

//对于构成圆弧的像素标记极性，如果梯度的方向和弧的方向指向一致，则为SAME_POLE,否则为OPP_POLE,该标记初始是为0
#define NOTDEF_POL 0
#define SAME_POL 1
#define OPP_POL  -1
/*----------------------------------------------------------------------------*/
/** Chained list of coordinates.
 */
struct coorlist
{
  int x,y;
  struct coorlist * next;
};
typedef struct ntuple_list_s
{
  int size;
  int max_size;
  int dim;
  double * values;
} * ntuple_list;

/*----------------------------------------------------------------------------*/
/** Free memory used in n-tuple 'in'.
 */
static void free_ntuple_list(ntuple_list in)
{
  if( in == NULL || in->values == NULL )
    error("free_ntuple_list: invalid n-tuple input.");
  free( (void *) in->values );
  free( (void *) in );
}

/*----------------------------------------------------------------------------*/
/** Create an n-tuple list and allocate memory for one element.
    @param dim the dimension (n) of the n-tuple.
 */
static ntuple_list new_ntuple_list(int dim)
{
  ntuple_list n_tuple;

  /* check parameters */
  if( dim == 0 ) error("new_ntuple_list: 'dim' must be positive.");

  /* get memory for list structure */
  n_tuple = (ntuple_list) malloc( sizeof(struct ntuple_list_s) );
  if( n_tuple == NULL ) error("not enough memory.");

  /* initialize list */
  n_tuple->size = 0;
  n_tuple->max_size = 1;
  n_tuple->dim = dim;

  /* get memory for tuples */
  n_tuple->values = (double *) malloc( dim*n_tuple->max_size * sizeof(double) );
  if( n_tuple->values == NULL ) error("not enough memory.");

  return n_tuple;
}

/*----------------------------------------------------------------------------*/
/** Enlarge the allocated memory of an n-tuple list.
 */
static void enlarge_ntuple_list(ntuple_list n_tuple)
{
  /* check parameters */
  if( n_tuple == NULL || n_tuple->values == NULL || n_tuple->max_size == 0 )
    error("enlarge_ntuple_list: invalid n-tuple.");

  /* duplicate number of tuples */
  n_tuple->max_size *= 2;

  /* realloc memory */
  n_tuple->values = (double *) realloc( (void *) n_tuple->values,
                      n_tuple->dim * n_tuple->max_size * sizeof(double) );
  if( n_tuple->values == NULL ) error("not enough memory.");
}

/*----------------------------------------------------------------------------*/
/** Add a 7-tuple to an n-tuple list.
 */
static void add_7tuple( ntuple_list out, double v1, double v2, double v3,
                        double v4, double v5, double v6, double v7 )
{
  /* check parameters */
  if( out == NULL ) error("add_7tuple: invalid n-tuple input.");
  if( out->dim != 7 ) error("add_7tuple: the n-tuple must be a 7-tuple.");

  /* if needed, alloc more tuples to 'out' */
  if( out->size == out->max_size ) enlarge_ntuple_list(out);
  if( out->values == NULL ) error("add_7tuple: invalid n-tuple input.");

  /* add new 7-tuple */
  out->values[ out->size * out->dim + 0 ] = v1;
  out->values[ out->size * out->dim + 1 ] = v2;
  out->values[ out->size * out->dim + 2 ] = v3;
  out->values[ out->size * out->dim + 3 ] = v4;
  out->values[ out->size * out->dim + 4 ] = v5;
  out->values[ out->size * out->dim + 5 ] = v6;
  out->values[ out->size * out->dim + 6 ] = v7;

  /* update number of tuples counter */
  out->size++;
}
/*----------------------------------------------------------------------------*/
/** Add a 8-tuple to an n-tuple list.
 */
static void add_8tuple( ntuple_list out, double v1, double v2, double v3,
                        double v4, double v5, double v6, double v7, int v8)
{
  /* check parameters */
  if( out == NULL ) error("add_8tuple: invalid n-tuple input.");
  if( out->dim != 8 ) error("add_8tuple: the n-tuple must be a 8-tuple.");

  /* if needed, alloc more tuples to 'out' */
  if( out->size == out->max_size ) enlarge_ntuple_list(out);
  if( out->values == NULL ) error("add_8tuple: invalid n-tuple input.");

  /* add new 8-tuple */
  out->values[ out->size * out->dim + 0 ] = v1;
  out->values[ out->size * out->dim + 1 ] = v2;
  out->values[ out->size * out->dim + 2 ] = v3;
  out->values[ out->size * out->dim + 3 ] = v4;
  out->values[ out->size * out->dim + 4 ] = v5;
  out->values[ out->size * out->dim + 5 ] = v6;
  out->values[ out->size * out->dim + 6 ] = v7;
  out->values[ out->size * out->dim + 7 ] = v8;

  /* update number of tuples counter */
  out->size++;
}
/** char image data type

    The pixel value at (x,y) is accessed by:

      image->data[ x + y * image->xsize ]

    with x and y integer.
 */
typedef struct image_char_s
{
  unsigned char * data;
  unsigned int xsize,ysize;
} * image_char;

/*----------------------------------------------------------------------------*/
/** Free memory used in image_char 'i'.
 */
static void free_image_char(image_char i)
{
  if( i == NULL || i->data == NULL )
    error("free_image_char: invalid input image.");
  free( (void *) i->data );
  free( (void *) i );
}

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize'.
 */
static image_char new_image_char(unsigned int xsize, unsigned int ysize)
{
  image_char image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 ) error("new_image_char: invalid image size.");

  /* get memory */
  image = (image_char) malloc( sizeof(struct image_char_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (unsigned char *) calloc( (size_t) (xsize*ysize),
                                          sizeof(unsigned char) );
  if( image->data == NULL ) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
static image_char new_image_char_ini( unsigned int xsize, unsigned int ysize,
                                      unsigned char fill_value )
{
  image_char image = new_image_char(xsize,ysize); /* create image */
  unsigned int N = xsize*ysize;
  unsigned int i;

  /* check parameters */
  if( image == NULL || image->data == NULL )
    error("new_image_char_ini: invalid image.");

  /* initialize */
  for(i=0; i<N; i++) image->data[i] = fill_value;

  return image;
}

/*----------------------------------------------------------------------------*/
/** int image data type

    The pixel value at (x,y) is accessed by:

      image->data[ x + y * image->xsize ]

    with x and y integer.
 */
typedef struct image_int_s
{
  int * data;
  unsigned int xsize,ysize;
} * image_int;

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize'.
 */
static image_int new_image_int(unsigned int xsize, unsigned int ysize)
{
  image_int image;

  /* check parameters */
  if( xsize == 0 || ysize == 0 ) error("new_image_int: invalid image size.");

  /* get memory */
  image = (image_int) malloc( sizeof(struct image_int_s) );
  if( image == NULL ) error("not enough memory.");
  image->data = (int *) calloc( (size_t) (xsize*ysize), sizeof(int) );
  if( image->data == NULL ) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
static image_int new_image_int_ini( unsigned int xsize, unsigned int ysize,
                                    int fill_value )
{
  image_int image = new_image_int(xsize,ysize); /* create image */
  unsigned int N = xsize*ysize;
  unsigned int i;

  /* initialize */
  for(i=0; i<N; i++) image->data[i] = fill_value;

  return image;
}
/** Compute a Gaussian kernel of length 'kernel->dim',
    standard deviation 'sigma', and centered at value 'mean'.

    For example, if mean=0.5, the Gaussian will be centered
    in the middle point2i between values 'kernel->values[0]'
    and 'kernel->values[1]'.
 */
static void gaussian_kernel(ntuple_list kernel, double sigma, double mean)
{
  double sum = 0.0;
  double val;
  int i;

  /* check parameters */
  if( kernel == NULL || kernel->values == NULL )
    error("gaussian_kernel: invalid n-tuple 'kernel'.");
  if( sigma <= 0.0 ) error("gaussian_kernel: 'sigma' must be positive.");

  /* compute Gaussian kernel */
  if( kernel->max_size < 1 ) enlarge_ntuple_list(kernel);
  kernel->size = 1;
  for(i=0;i<kernel->dim;i++)
    {
      val = ( (double) i - mean ) / sigma;
      kernel->values[i] = exp( -0.5 * val * val );
      sum += kernel->values[i];
    }

  /* normalization */
  if( sum >= 0.0 ) for(i=0;i<kernel->dim;i++) kernel->values[i] /= sum;
}

/*----------------------------------------------------------------------------*/
/** Scale the input image 'in' by a factor 'scale' by Gaussian sub-sampling.

    For example, scale=0.8 will give a result at 80% of the original size.

    The image is convolved with a Gaussian kernel
    @f[
        G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}
    @f]
    before the sub-sampling to prevent aliasing.

    The standard deviation sigma given by:
    -  sigma = sigma_scale / scale,   if scale <  1.0
    -  sigma = sigma_scale,           if scale >= 1.0

    To be able to sub-sample at non-integer steps, some interpolation
    is needed. In this implementation, the interpolation is done by
    the Gaussian kernel, so both operations (filtering and sampling)
    are done at the same time. The Gaussian kernel is computed
    centered on the coordinates of the required sample. In this way,
    when applied, it gives directly the result of convolving the image
    with the kernel and interpolated to that particular position.

    A fast algorithm is done using the separability of the Gaussian
    kernel. Applying the 2D Gaussian kernel is equivalent to applying
    first a horizontal 1D Gaussian kernel and then a vertical 1D
    Gaussian kernel (or the other way round). The reason is that
    @f[
        G(x,y) = G(x) * G(y)
    @f]
    where
    @f[
        G(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}}.
    @f]
    The algorithm first applies a combined Gaussian kernel and sampling
    in the x axis, and then the combined Gaussian kernel and sampling
    in the y axis.
 */
static image_double gaussian_sampler( image_double in, double scale,
                                      double sigma_scale )
{
  image_double aux,out;
  ntuple_list kernel;
  int N,M,h,n,x,y,i;
  int xc,yc,j,double_x_size,double_y_size;
  double sigma,xx,yy,sum,prec;

  /* check parameters */
  if( in == NULL || in->data == NULL || in->xsize == 0 || in->ysize == 0 )
    error("gaussian_sampler: invalid image.");
  if( scale <= 0.0 ) error("gaussian_sampler: 'scale' must be positive.");
  if( sigma_scale <= 0.0 )
    error("gaussian_sampler: 'sigma_scale' must be positive.");

  /* compute new image size and get memory for images */
  if( in->xsize * scale > (double) UINT_MAX ||
      in->ysize * scale > (double) UINT_MAX )
    error("gaussian_sampler: the output image size exceeds the handled size.");
  N = (unsigned int) ceil( in->xsize * scale );//上取整
  M = (unsigned int) ceil( in->ysize * scale );
  aux = new_image_double(N,in->ysize);
  out = new_image_double(N,M);

  /* sigma, kernel size and memory for the kernel */
  sigma = scale < 1.0 ? sigma_scale / scale : sigma_scale;
  /*
     The size of the kernel is selected to guarantee that the
     the first discarded term is at least 10^prec times smaller
     than the central value. For that, h should be larger than x, with
       e^(-x^2/2sigma^2) = 1/10^prec.
     Then,
       x = sigma * sqrt( 2 * prec * ln(10) ).
   */
  prec = 3.0;//高斯核的最外围降到10^(-3)
  h = (unsigned int) ceil( sigma * sqrt( 2.0 * prec * log(10.0) ) );
  n = 1+2*h; /* kernel size */
  kernel = new_ntuple_list(n);

  /* auxiliary double image size variables */
  double_x_size = (int) (2 * in->xsize);
  double_y_size = (int) (2 * in->ysize);

  /* First subsampling: x axis */
  for(x=0;x<aux->xsize;x++)
    {
      /*
         x   is the coordinate in the new image.
         xx  is the corresponding x-value in the original size image.
         xc  is the integer value, the pixel coordinate of xx.
       */
      xx = (double) x / scale;
      /* coordinate (0.0,0.0) is in the center of pixel (0,0),
         so the pixel with xc=0 get the values of xx from -0.5 to 0.5 */
      xc = (int) floor( xx + 0.5 );
      gaussian_kernel( kernel, sigma, (double) h + xx - (double) xc );
      /* the kernel must be computed for each x because the fine
         offset xx-xc is different in each case */

      for(y=0;y<aux->ysize;y++)
        {
          sum = 0.0;
          for(i=0;i<kernel->dim;i++)
            {
              j = xc - h + i;

              /* symmetry boundary condition */
              while( j < 0 ) j += double_x_size;
              while( j >= double_x_size ) j -= double_x_size;
              if( j >= (int) in->xsize ) j = double_x_size-1-j;

              sum += in->data[ j + y * in->xsize ] * kernel->values[i];
            }
          aux->data[ x + y * aux->xsize ] = sum;
        }
    }

  /* Second subsampling: y axis */
  for(y=0;y<out->ysize;y++)
    {
      /*
         y   is the coordinate in the new image.
         yy  is the corresponding x-value in the original size image.
         yc  is the integer value, the pixel coordinate of xx.
       */
      yy = (double) y / scale;
      /* coordinate (0.0,0.0) is in the center of pixel (0,0),
         so the pixel with yc=0 get the values of yy from -0.5 to 0.5 */
      yc = (int) floor( yy + 0.5 );
      gaussian_kernel( kernel, sigma, (double) h + yy - (double) yc );
      /* the kernel must be computed for each y because the fine
         offset yy-yc is different in each case */

      for(x=0;x<out->xsize;x++)
        {
          sum = 0.0;
          for(i=0;i<kernel->dim;i++)
            {
              j = yc - h + i;

              /* symmetry boundary condition */
              while( j < 0 ) j += double_y_size;
              while( j >= double_y_size ) j -= double_y_size;
              if( j >= (int) in->ysize ) j = double_y_size-1-j;

              sum += aux->data[ x + j * aux->xsize ] * kernel->values[i];
            }
          out->data[ x + y * out->xsize ] = sum;
        }
    }

  /* free memory */
  free_ntuple_list(kernel);
  free_image_double(aux);

  return out;
}


/*----------------------------------------------------------------------------*/
/*--------------------------------- Gradient ---------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Computes the direction of the level line of 'in' at each point2i.

    The result is:
    - an image_double with the angle at each pixel, or NOTDEF if not defined.
    - the image_double 'modgrad' (a point2ier is passed as argument)
      with the gradient magnitude at each point2i.
    - a list of pixels 'list_p' roughly ordered by decreasing
      gradient magnitude. (The order is made by classifying point2is
      into bins by gradient magnitude. The parameters 'n_bins' and
      'max_grad' specify the number of bins and the gradient modulus
      at the highest bin. The pixels in the list would be in
      decreasing gradient magnitude, up to a precision of the size of
      the bins.)
    - a point2ier 'mem_p' to the memory used by 'list_p' to be able to
      free the memory when it is not used anymore.
 */
//返回一张梯度角度顺时针旋转90°后的align角度图angles，如果梯度角度是(gx,gy)->(-gy,gx)，
//和梯度的模的图modgrad,然后按照n_bins进行伪排序返回链表的头指针list_p,里面存的是坐标
static image_double ll_angle( image_double in, double threshold,
                              struct coorlist ** list_p,
                              image_double * modgrad, unsigned int n_bins )
{
  image_double g;
  unsigned int n,p,x,y,adr,i;
  double com1,com2,gx,gy,norm,norm2;
  /* the rest of the variables are used for pseudo-ordering
     the gradient magnitude values */
  int list_count = 0;
  //struct coorlist * list;
  struct coorlist *temp;
  struct coorlist ** range_l_s; /* array of point2iers to start of bin list,表示1024个bin的头指针的指针数组 */
  struct coorlist ** range_l_e; /* array of point2iers to end of bin list，表示1024个bin的尾指针的指针数组*/
  struct coorlist * start;
  struct coorlist * end;
  double max_grad = 0.0;

  /* check parameters */
  if( in == NULL || in->data == NULL || in->xsize == 0 || in->ysize == 0 )
    error("ll_angle: invalid image.");
  if( threshold < 0.0 ) error("ll_angle: 'threshold' must be positive.");
  if( list_p == NULL ) error("ll_angle: NULL point2ier 'list_p'.");
 // if( mem_p == NULL ) error("ll_angle: NULL point2ier 'mem_p'.");
  if( modgrad == NULL ) error("ll_angle: NULL point2ier 'modgrad'.");
  if( n_bins == 0 ) error("ll_angle: 'n_bins' must be positive.");

  /* image size shortcuts */
  n = in->ysize;
  p = in->xsize;

  /* allocate output image */
  g = new_image_double(in->xsize,in->ysize);

  /* get memory for the image of gradient modulus */
  *modgrad = new_image_double(in->xsize,in->ysize);

  /* get memory for "ordered" list of pixels */
  //list = (struct coorlist *) calloc( (size_t) (n*p), sizeof(struct coorlist) );
  //*mem_p = (void *) list;
  range_l_s = (struct coorlist **) calloc( (size_t) n_bins,
                                           sizeof(struct coorlist *) );
  range_l_e = (struct coorlist **) calloc( (size_t) n_bins,
                                           sizeof(struct coorlist *) );
 // if( list == NULL || range_l_s == NULL || range_l_e == NULL )
  if( range_l_s == NULL || range_l_e == NULL )
    error("not enough memory.");
  for(i=0;i<n_bins;i++) range_l_s[i] = range_l_e[i] = NULL;

  /* 'undefined' on the down and right boundaries */
  for(x=0;x<p;x++) g->data[(n-1)*p+x] = NOTDEF;// p = in->xsize
  for(y=0;y<n;y++) g->data[p*y+p-1]   = NOTDEF;// n = in->ysize;

  /* compute gradient on the remaining pixels */
  for(x=0;x<p-1;x++)
    for(y=0;y<n-1;y++)
      {
        adr = y*p+x;

        /*
           Norm 2 computation using 2x2 pixel window:
             A B
             C D
           and
             com1 = D-A,  com2 = B-C.
           Then
             gx = B+D - (A+C)   horizontal difference
             gy = C+D - (A+B)   vertical difference
           com1 and com2 are just to avoid 2 additions.
         */
        com1 = in->data[adr+p+1] - in->data[adr];
        com2 = in->data[adr+1]   - in->data[adr+p];

        gx = com1+com2; /* gradient x component */
        gy = com1-com2; /* gradient y component */
        norm2 = gx*gx+gy*gy;
        norm = sqrt( norm2 / 4.0 ); /* gradient norm */

        (*modgrad)->data[adr] = norm; /* store gradient norm */

        if( norm <= threshold ) /* norm too small, gradient no defined */
          g->data[adr] = NOTDEF; /* gradient angle not defined */
        else
          {
            /* gradient angle computation */
            g->data[adr] = atan2(gx,-gy);

            /* look for the maximum of the gradient */
            if( norm > max_grad ) max_grad = norm;
          }
      }

  /* compute histogram of gradient values */
  for(x=0;x<p-1;x++)
    for(y=0;y<n-1;y++)
      {
		temp = new coorlist();
		if(temp == NULL)
		{
			printf("not enough memory");
			system("pause");
		}
        norm = (*modgrad)->data[y*p+x];
        /* store the point2i in the right bin according to its norm */
        i = (unsigned int) (norm * (double) n_bins / max_grad);
        if( i >= n_bins ) i = n_bins-1;
        if( range_l_e[i] == NULL )
          range_l_s[i] = range_l_e[i] = temp;//记录第i个区域的头指针到range_l_s[i]
        else
          {
            range_l_e[i]->next = temp;//第i个区域由尾指针range_l_e[i]完成勾链
            range_l_e[i] = temp;
          }
        range_l_e[i]->x = (int) x;//将坐标(x,y)记录到第i个分区
        range_l_e[i]->y = (int) y;
        range_l_e[i]->next = NULL;
      }

  /* Make the list of pixels (almost) ordered by norm value.
     It starts by the larger bin, so the list starts by the
     pixels with the highest gradient value. Pixels would be ordered
     by norm value, up to a precision given by max_grad/n_bins.
   */
  for(i=n_bins-1; i>0 && range_l_s[i]==NULL; i--);//找到第一个不为空的分区bin
  start = range_l_s[i];
  end = range_l_e[i];
  if( start != NULL )
    while(i>0)
      {
        --i;
        if( range_l_s[i] != NULL )
          {
            end->next = range_l_s[i];
            end = range_l_e[i];
          }
      }
  *list_p = start;
 // *mem_p  = start;
  /* free memory */
  free( (void *) range_l_s );
  free( (void *) range_l_e );

  return g;
}
/*----------------------------------------------------------------------------*/
/** Is point2i (x,y) aligned to angle theta, up to precision 'prec'?
 */
static int isaligned( int x, int y, image_double angles, double theta,
                      double prec )
{
  double a;

  /* check parameters */
  if( angles == NULL || angles->data == NULL )
    error("isaligned: invalid image 'angles'.");
  if( x < 0 || y < 0 || x >= (int) angles->xsize || y >= (int) angles->ysize )
    error("isaligned: (x,y) out of the image.");
  if( prec < 0.0 ) error("isaligned: 'prec' must be positive.");

  /* angle at pixel (x,y) */
  a = angles->data[ x + y * angles->xsize ];

  /* pixels whose level-line angle is not defined
     are considered as NON-aligned */
  if( a == NOTDEF ) return FALSE;  /* there is no need to call the function
                                      'double_equal' here because there is
                                      no risk of problems related to the
                                      comparison doubles, we are only
                                      interested in the exact NOTDEF value */

  /* it is assumed that 'theta' and 'a' are in the range [-pi,pi] */
  theta -= a;
  if( theta < 0.0 ) theta = -theta;
  if( theta > M_3_2_PI )
    {
	//--------------------------------------
	//origin code
     /* theta -= M_2__PI;
      if( theta < 0.0 ) theta = -theta;*/
	//--------------------------------------
	  //-------------------------------------
	  //mycode
	  theta = M_2__PI-theta;
	  if(theta < 0.0) 
		 theta = -theta; 
	  //--------------------------------------
    }

  return theta <= prec;
}


/*----------------------------------------------------------------------------*/
/*----------------------------- NFA computation ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x using the Lanczos approximation.
    See http://www.rskey.org/gamma.htm

    The formula used is
    @f[
      \Gamma(x) = \frac{ \sum_{n=0}^{N} q_n x^n }{ \Pi_{n=0}^{N} (x+n) }
                  (x+5.5)^{x+0.5} e^{-(x+5.5)}
    @f]
    so
    @f[
      \log\Gamma(x) = \log\left( \sum_{n=0}^{N} q_n x^n \right)
                      + (x+0.5) \log(x+5.5) - (x+5.5) - \sum_{n=0}^{N} \log(x+n)
    @f]
    and
      q0 = 75122.6331530,
      q1 = 80916.6278952,
      q2 = 36308.2951477,
      q3 = 8687.24529705,
      q4 = 1168.92649479,
      q5 = 83.8676043424,
      q6 = 2.50662827511.
 */
static double log_gamma_lanczos(double x)
{
  static double q[7] = { 75122.6331530, 80916.6278952, 36308.2951477,
                         8687.24529705, 1168.92649479, 83.8676043424,
                         2.50662827511 };
  double a = (x+0.5) * log(x+5.5) - (x+5.5);
  double b = 0.0;
  int n;

  for(n=0;n<7;n++)
    {
      a -= log( x + (double) n );
      b += q[n] * pow( x, (double) n );
    }
  return a + log(b);
}

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x using Windschitl method.
    See http://www.rskey.org/gamma.htm

    The formula used is
    @f[
        \Gamma(x) = \sqrt{\frac{2\pi}{x}} \left( \frac{x}{e}
                    \sqrt{ x\sinh(1/x) + \frac{1}{810x^6} } \right)^x
    @f]
    so
    @f[
        \log\Gamma(x) = 0.5\log(2\pi) + (x-0.5)\log(x) - x
                      + 0.5x\log\left( x\sinh(1/x) + \frac{1}{810x^6} \right).
    @f]
    This formula is a good approximation when x > 15.
 */
static double log_gamma_windschitl(double x)
{
  return 0.918938533204673 + (x-0.5)*log(x) - x
         + 0.5*x*log( x*sinh(1/x) + 1/(810.0*pow(x,6.0)) );
}

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x. When x>15 use log_gamma_windschitl(),
    otherwise use log_gamma_lanczos().
 */
#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

/*----------------------------------------------------------------------------*/
/** Size of the table to store already computed inverse values.
 */
#define TABSIZE 100000

/*----------------------------------------------------------------------------*/
/** Computes -log10(NFA).

    NFA stands for Number of False Alarms:
    @f[
        \mathrm{NFA} = NT \cdot B(n,k,p)
    @f]

    - NT       - number of tests
    - B(n,k,p) - tail of binomial distribution with parameters n,k and p:
    @f[
        B(n,k,p) = \sum_{j=k}^n
                   \left(\begin{array}{c}n\\j\end{array}\right)
                   p^{j} (1-p)^{n-j}
    @f]

    The value -log10(NFA) is equivalent but more intuitive than NFA:
    - -1 corresponds to 10 mean false alarms
    -  0 corresponds to 1 mean false alarm
    -  1 corresponds to 0.1 mean false alarms
    -  2 corresponds to 0.01 mean false alarms
    -  ...

    Used this way, the bigger the value, better the detection,
    and a logarithmic scale is used.

    @param n,k,p binomial parameters.
    @param logNT logarithm of Number of Tests

    The computation is based in the gamma function by the following
    relation:
    @f[
        \left(\begin{array}{c}n\\k\end{array}\right)
        = \frac{ \Gamma(n+1) }{ \Gamma(k+1) \cdot \Gamma(n-k+1) }.
    @f]
    We use efficient algorithms to compute the logarithm of
    the gamma function.

    To make the computation faster, not all the sum is computed, part
    of the terms are neglected based on a bound to the error obtained
    (an error of 10% in the result is accepted).
 */
static double nfa(int n, int k, double p, double logNT)
{
  static double inv[TABSIZE];   /* table to keep computed inverse values */
  double tolerance = 0.1;       /* an error of 10% in the result is accepted */
  double log1term,term,bin_term,mult_term,bin_tail,err,p_term;
  int i;

  /* check parameters */
  if( n<0 || k<0 || k>n || p<=0.0 || p>=1.0 )
    error("nfa: wrong n, k or p values.");

  /* trivial cases */
  if( n==0 || k==0 ) return -logNT;
  if( n==k ) return -logNT - (double) n * log10(p);

  /* probability term */
  p_term = p / (1.0-p);

  /* compute the first term of the series */
  /*
     binomial_tail(n,k,p) = sum_{i=k}^n bincoef(n,i) * p^i * (1-p)^{n-i}
     where bincoef(n,i) are the binomial coefficients.
     But
       bincoef(n,k) = gamma(n+1) / ( gamma(k+1) * gamma(n-k+1) ).
     We use this to compute the first term. Actually the log of it.
   */
  log1term = log_gamma( (double) n + 1.0 ) - log_gamma( (double) k + 1.0 )
           - log_gamma( (double) (n-k) + 1.0 )
           + (double) k * log(p) + (double) (n-k) * log(1.0-p);
  term = exp(log1term);

  /* in some cases no more computations are needed */
  if( double_equal(term,0.0) )              /* the first term is almost zero */
    {
      if( (double) k > (double) n * p )     /* at begin or end of the tail?  */
        return -log1term / M_LN10 - logNT;  /* end: use just the first term  */
      else
        return -logNT;                      /* begin: the tail is roughly 1  */
    }

  /* compute more terms if needed */
  bin_tail = term;
  for(i=k+1;i<=n;i++)
    {
      /*
         As
           term_i = bincoef(n,i) * p^i * (1-p)^(n-i)
         and
           bincoef(n,i)/bincoef(n,i-1) = n-1+1 / i,
         then,
           term_i / term_i-1 = (n-i+1)/i * p/(1-p)
         and
           term_i = term_i-1 * (n-i+1)/i * p/(1-p).
         1/i is stored in a table as they are computed,
         because divisions are expensive.
         p/(1-p) is computed only once and stored in 'p_term'.
       */
      bin_term = (double) (n-i+1) * ( i<TABSIZE ?
                   ( inv[i]!=0.0 ? inv[i] : ( inv[i] = 1.0 / (double) i ) ) :
                   1.0 / (double) i );

      mult_term = bin_term * p_term;
      term *= mult_term;
      bin_tail += term;
      if(bin_term<1.0)
        {
          /* When bin_term<1 then mult_term_j<mult_term_i for j>i.
             Then, the error on the binomial tail when truncated at
             the i term can be bounded by a geometric series of form
             term_i * sum mult_term_i^j.                            */
          err = term * ( ( 1.0 - pow( mult_term, (double) (n-i+1) ) ) /
                         (1.0-mult_term) - 1.0 );

          /* One wants an error at most of tolerance*final_result, or:
             tolerance * abs(-log10(bin_tail)-logNT).
             Now, the error that can be accepted on bin_tail is
             given by tolerance*final_result divided by the derivative
             of -log10(x) when x=bin_tail. that is:
             tolerance * abs(-log10(bin_tail)-logNT) / (1/bin_tail)
             Finally, we truncate the tail if the error is less than:
             tolerance * abs(-log10(bin_tail)-logNT) * bin_tail        */
          if( err < tolerance * fabs(-log10(bin_tail)-logNT) * bin_tail ) break;
        }
    }
  double nfavalue = -log10(bin_tail) - logNT;
  return nfavalue;
}

/*----------------------------------------------------------------------------*/
/** Compute a rectangle's NFA value.
 */
static double rect_nfa(struct rect * rec, image_double angles, double logNT)
{
  rect_iter * i;
  int pts = 0;
  int alg = 0;

  /* check parameters */
  if( rec == NULL ) error("rect_nfa: invalid rectangle.");
  if( angles == NULL ) error("rect_nfa: invalid 'angles'.");

  /* compute the total number of pixels and of aligned point2is in 'rec' */
  for(i=ri_ini(rec); !ri_end(i); ri_inc(i)) /* rectangle iterator */
    if( i->x >= 0 && i->y >= 0 &&
        i->x < (int) angles->xsize && i->y < (int) angles->ysize )
      {
        ++pts; /* total number of pixels counter */
        if( isaligned(i->x, i->y, angles, rec->theta, rec->prec) )
          ++alg; /* aligned point2is counter */
      }
  ri_del(i); /* delete iterator */
  double NFAvalue = nfa(pts,alg,rec->p,logNT); /* compute NFA value */
  return NFAvalue;
}
/*---------------------------------- Regions ---------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Compute region's angle as the principal inertia axis of the region.

    The following is the region inertia matrix A:
    @f[

        A = \left(\begin{array}{cc}
                                    Ixx & Ixy \\
                                    Ixy & Iyy \\
             \end{array}\right)

    @f]
    where

      Ixx =   sum_i G(i).(y_i - cx)^2

      Iyy =   sum_i G(i).(x_i - cy)^2

      Ixy = - sum_i G(i).(x_i - cx).(y_i - cy)

    and
    - G(i) is the gradient norm at pixel i, used as pixel's weight.
    - x_i and y_i are the coordinates of pixel i.
    - cx and cy are the coordinates of the center of th region.

    lambda1 and lambda2 are the eigenvalues of matrix A,
    with lambda1 >= lambda2. They are found by solving the
    characteristic polynomial:

      det( lambda I - A) = 0

    that gives:

      lambda1 = ( Ixx + Iyy + sqrt( (Ixx-Iyy)^2 + 4.0*Ixy*Ixy) ) / 2

      lambda2 = ( Ixx + Iyy - sqrt( (Ixx-Iyy)^2 + 4.0*Ixy*Ixy) ) / 2

    To get the line segment direction we want to get the angle the
    eigenvector associated to the smallest eigenvalue. We have
    to solve for a,b in:

      a.Ixx + b.Ixy = a.lambda2

      a.Ixy + b.Iyy = b.lambda2

    We want the angle theta = atan(b/a). It can be computed with
    any of the two equations:

      theta = atan( (lambda2-Ixx) / Ixy )

    or

      theta = atan( Ixy / (lambda2-Iyy) )

    When |Ixx| > |Iyy| we use the first, otherwise the second (just to
    get better numeric precision).
 */
static double get_theta( point2i * reg, int reg_size, double x, double y,
                         image_double modgrad, double reg_angle, double prec )
{
  double lambda,theta,weight;
  double Ixx = 0.0;
  double Iyy = 0.0;
  double Ixy = 0.0;
  double temp1,temp2;
  int i;

  /* check parameters */
  if( reg == NULL ) error("get_theta: invalid region.");
  if( reg_size <= 1 ) error("get_theta: region size <= 1.");
  if( modgrad == NULL || modgrad->data == NULL )
    error("get_theta: invalid 'modgrad'.");
  if( prec < 0.0 ) error("get_theta: 'prec' must be positive.");

  /* compute inertia matrix */
  for(i=0; i<reg_size; i++)
    {
      weight = modgrad->data[ reg[i].x + reg[i].y * modgrad->xsize ];
      Ixx += ( (double) reg[i].y - y ) * ( (double) reg[i].y - y ) * weight;
      Iyy += ( (double) reg[i].x - x ) * ( (double) reg[i].x - x ) * weight;
      Ixy -= ( (double) reg[i].x - x ) * ( (double) reg[i].y - y ) * weight;
    }
  if( double_equal(Ixx,0.0) && double_equal(Iyy,0.0) && double_equal(Ixy,0.0) )//判断Ixx、Iyy、Ixy与0是否非常接近，由于它们为double类型，故需要专门的函数判断
    error("get_theta: null inertia matrix.");

  /* compute smallest eigenvalue */
  lambda = 0.5 * ( Ixx + Iyy - sqrt( (Ixx-Iyy)*(Ixx-Iyy) + 4.0*Ixy*Ixy ) );

  /* compute angle */
  theta = fabs(Ixx)>fabs(Iyy) ? atan2(lambda-Ixx,Ixy) : atan2(Ixy,lambda-Iyy);
  /* The previous procedure doesn't cares about orientation,
     so it could be wrong by 180 degrees. Here is corrected if necessary. */
  temp1 = angle_diff(theta,reg_angle);
  if( temp1 > prec )//这是由于用惯性矩阵算出的两个正交轴的较小特征值对应的角度和该区域的角度可能相差180°
  {
	  //------------------------------------------
	  //theta += M_PI;   //origin code
	  //------------------------------------------
	  //------------------------------------------
	  //my code,增加该段代码，限制theta在 (-pi,pi)之间
	  //int flag = 0;
	  temp2 = angle_diff(theta+M_PI,reg_angle);
	  if(temp2 < prec)
	  {
		  theta += M_PI;
		if(theta > M_PI)
		{
		   theta -= M_2__PI;
		   //flag = 1;
		   //if(angle_diff(theta,reg_angle) > prec)
		   //{
		   //	  //flag = 2;
		   //	  theta = reg_angle;
		   // }
		}
	  }
	  else
	  {
		  theta = (temp2 <= temp1) ? (theta+M_PI) : theta;
		  while( theta <= -M_PI ) theta += M_2__PI;
          while( theta >   M_PI ) theta -= M_2__PI;
	  }
	  
	  //--------------------------------------------
  }
  return theta;
}

/*----------------------------------------------------------------------------*/
/** Computes a rectangle that covers a region of point2is.
 */
static void region2rect( point2i * reg, int reg_size,
						image_double modgrad, double reg_angle,
                         double prec, double p, struct rect * rec )
{
  double x,y,dx,dy,l,w,theta,weight,sum,l_min,l_max,w_min,w_max;
  int i;

  /* check parameters */
  if( reg == NULL ) error("region2rect: invalid region.");
  if( reg_size <= 1 ) error("region2rect: region size <= 1.");
  if( modgrad == NULL || modgrad->data == NULL )
    error("region2rect: invalid image 'modgrad'.");
  if( rec == NULL ) error("region2rect: invalid 'rec'.");

  /* center of the region:

     It is computed as the weighted sum of the coordinates
     of all the pixels in the region. The norm of the gradient
     is used as the weight of a pixel. The sum is as follows:
       cx = \sum_i G(i).x_i
       cy = \sum_i G(i).y_i
     where G(i) is the norm of the gradient of pixel i
     and x_i,y_i are its coordinates.
   */
  //获得质心 x,y
  x = y = sum = 0.0;
  for(i=0; i<reg_size; i++)
    {
      weight = modgrad->data[ reg[i].x + reg[i].y * modgrad->xsize ];
      x += (double) reg[i].x * weight;
      y += (double) reg[i].y * weight;
      sum += weight;
    }
  if( sum <= 0.0 ) error("region2rect: weights sum equal to zero.");
  x /= sum;
  y /= sum;

  /* theta */
  //运用惯性矩阵获得更为精确的角度估计
  theta = get_theta(reg,reg_size,x,y,modgrad,reg_angle,prec);
  dx = cos(theta);
  dy = sin(theta);

  /* length and width:

     'l' and 'w' are computed as the distance from the center of the
     region to pixel i, projected along the rectangle axis (dx,dy) and
     to the orthogonal axis (-dy,dx), respectively.

     The length of the rectangle goes from l_min to l_max, where l_min
     and l_max are the minimum and maximum values of l in the region.
     Analogously, the width is selected from w_min to w_max, where
     w_min and w_max are the minimum and maximum of w for the pixels
     in the region.
   */
  //因为区域的方向向量为 (dx,dy) 
  /*
  ------------------->x
  |\
  | \  
  |  \(dx,dy)
  |   
 \|/
  y
  因此顺时针旋转90°是 (-dy,dx)
  */
  l_min = l_max = w_min = w_max = 0.0;
  for(i=0; i<reg_size; i++)//用向量内积求在线段方向和与线段方向垂直方向的投影求l,w
    {
      l =  ( (double) reg[i].x - x) * dx + ( (double) reg[i].y - y) * dy;
      w = -( (double) reg[i].x - x) * dy + ( (double) reg[i].y - y) * dx;

      if( l > l_max ) l_max = l;
      if( l < l_min ) l_min = l;
      if( w > w_max ) w_max = w;
      if( w < w_min ) w_min = w;
    }

  /* store values */
  rec->x1 = x + l_min * dx;
  rec->y1 = y + l_min * dy;
  rec->x2 = x + l_max * dx;
  rec->y2 = y + l_max * dy;
  rec->width = w_max - w_min;
  rec->x = x;
  rec->y = y;
  rec->theta = theta;
  rec->dx = dx;
  rec->dy = dy;
  rec->prec = prec;
  rec->p = p;

  /* we impose a minimal width of one pixel

     A sharp horizontal or vertical step would produce a perfectly
     horizontal or vertical region. The width computed would be
     zero. But that corresponds to a one pixels width transition in
     the image.
   */
  if( rec->width < 1.0 ) 
	  rec->width = 1.0;
}

//区域质心和角度已经计算好了，因此只进行矩形近似。而region2rect此外还进行了质心和角度计算。
static void region2rect2(point2i * reg, int reg_size,double reg_center_x,double reg_center_y,
					double reg_theta,double prec, double p, struct rect * rec )
{
  double dx,dy,l,w,l_min,l_max,w_min,w_max;
  int i;
  /* check parameters */
  if( reg == NULL ) error("region2rect: invalid region.");
  if( reg_size <= 1 ) error("region2rect: region size <= 1.");
  if( rec == NULL ) error("region2rect: invalid 'rec'.");

  //获得区域的方向向量(dx,dy)
  dx = cos(reg_theta);
  dy = sin(reg_theta);
  l_min = l_max = w_min = w_max = 0.0;
  for(i=0; i<reg_size; i++)//用向量内积求在线段方向和与线段方向垂直方向的投影求l,w
    {
      l =  ( (double) reg[i].x - reg_center_x) * dx + ( (double) reg[i].y - reg_center_y) * dy;
      w = -( (double) reg[i].x - reg_center_x) * dy + ( (double) reg[i].y - reg_center_y) * dx;

      if( l > l_max ) l_max = l;
      if( l < l_min ) l_min = l;
      if( w > w_max ) w_max = w;
      if( w < w_min ) w_min = w;
    }

  /* store values */
  rec->x1 = reg_center_x + l_min * dx;
  rec->y1 = reg_center_y + l_min * dy;
  rec->x2 = reg_center_x + l_max * dx;
  rec->y2 = reg_center_y + l_max * dy;
  rec->width = w_max - w_min;
  rec->x = reg_center_x;
  rec->y = reg_center_y;
  rec->theta = reg_theta;
  rec->dx = dx;
  rec->dy = dy;
  rec->prec = prec;
  rec->p = p;

  /* we impose a minimal width of one pixel

     A sharp horizontal or vertical step would produce a perfectly
     horizontal or vertical region. The width computed would be
     zero. But that corresponds to a one pixels width transition in
     the image.
   */
  if( rec->width < 1.0 ) 
	 rec->width = 1.0;
}
/*----------------------------------------------------------------------------*/
/** Build a region of pixels that share the same angle, up to a
    tolerance 'prec', starting at point2i (x,y).
 */
static void region_grow( int x, int y, image_double angles, struct point2i * reg,
                         int * reg_size, double * reg_angle, image_char used,
                         double prec )
{
  double sumdx,sumdy;
  int xx,yy,i; 

  /* check parameters */
  if( x < 0 || y < 0 || x >= (int) angles->xsize || y >= (int) angles->ysize )
    error("region_grow: (x,y) out of the image.");
  if( angles == NULL || angles->data == NULL )
    error("region_grow: invalid image 'angles'.");
  if( reg == NULL ) error("region_grow: invalid 'reg'.");
  if( reg_size == NULL ) error("region_grow: invalid point2ier 'reg_size'.");
  if( reg_angle == NULL ) error("region_grow: invalid point2ier 'reg_angle'.");
  if( used == NULL || used->data == NULL )
    error("region_grow: invalid image 'used'.");

  /* first point2i of the region */
  *reg_size = 1;
  reg[0].x = x;
  reg[0].y = y;
  *reg_angle = angles->data[x+y*angles->xsize];  /* region's angle */
  sumdx = cos(*reg_angle);
  sumdy = sin(*reg_angle);
  used->data[x+y*used->xsize] = USED;

  /* try neighbors as new region point2is */
  for(i=0; i<*reg_size; i++)
    for(xx=reg[i].x-1; xx<=reg[i].x+1; xx++)
      for(yy=reg[i].y-1; yy<=reg[i].y+1; yy++)
        if( xx>=0 && yy>=0 && xx<(int)used->xsize && yy<(int)used->ysize &&
            used->data[xx+yy*used->xsize] != USED &&
            isaligned(xx,yy,angles,*reg_angle,prec) )
          {
            /* add point2i */
            used->data[xx+yy*used->xsize] = USED;
            reg[*reg_size].x = xx;
            reg[*reg_size].y = yy;
            ++(*reg_size);

            /* update region's angle */
            sumdx += cos( angles->data[xx+yy*angles->xsize] );
            sumdy += sin( angles->data[xx+yy*angles->xsize] );
            *reg_angle = atan2(sumdy,sumdx);
          }
}

/*----------------------------------------------------------------------------*/
/** Try some rectangles variations to improve NFA value. Only if the
    rectangle is not meaningful (i.e., log_nfa <= log_eps).
 */
static double rect_improve( struct rect * rec, image_double angles,
                            double logNT, double log_eps )
{
  struct rect r;
  double log_nfa,log_nfa_new;
  double delta = 0.5;
  double delta_2 = delta / 2.0;
  int n;

  log_nfa = rect_nfa(rec,angles,logNT);

  if( log_nfa > log_eps ) return log_nfa;

  /* try finer precisions */
  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      r.p /= 2.0;
      r.prec = r.p * M_PI;
      log_nfa_new = rect_nfa(&r,angles,logNT);
      if( log_nfa_new > log_nfa )
        {
          log_nfa = log_nfa_new;
          rect_copy(&r,rec);
        }
    }

  if( log_nfa > log_eps ) return log_nfa;

  /* try to reduce width */
  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      if( (r.width - delta) >= 0.5 )
        {
          r.width -= delta;
          log_nfa_new = rect_nfa(&r,angles,logNT);
          if( log_nfa_new > log_nfa )
            {
              rect_copy(&r,rec);
              log_nfa = log_nfa_new;
            }
        }
    }

  if( log_nfa > log_eps ) return log_nfa;

  /* try to reduce one side of the rectangle */
  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      if( (r.width - delta) >= 0.5 )
        {
          r.x1 += -r.dy * delta_2;
          r.y1 +=  r.dx * delta_2;
          r.x2 += -r.dy * delta_2;
          r.y2 +=  r.dx * delta_2;
          r.width -= delta;
          log_nfa_new = rect_nfa(&r,angles,logNT);
          if( log_nfa_new > log_nfa )
            {
              rect_copy(&r,rec);
              log_nfa = log_nfa_new;
            }
        }
    }

  if( log_nfa > log_eps ) return log_nfa;

  /* try to reduce the other side of the rectangle */
  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      if( (r.width - delta) >= 0.5 )
        {
          r.x1 -= -r.dy * delta_2;
          r.y1 -=  r.dx * delta_2;
          r.x2 -= -r.dy * delta_2;
          r.y2 -=  r.dx * delta_2;
          r.width -= delta;
          log_nfa_new = rect_nfa(&r,angles,logNT);
          if( log_nfa_new > log_nfa )
            {
              rect_copy(&r,rec);
              log_nfa = log_nfa_new;
            }
        }
    }

  if( log_nfa > log_eps ) return log_nfa;

  /* try even finer precisions */
  rect_copy(rec,&r);
  for(n=0; n<5; n++)
    {
      r.p /= 2.0;
      r.prec = r.p * M_PI;
      log_nfa_new = rect_nfa(&r,angles,logNT);
      if( log_nfa_new > log_nfa )
        {
          log_nfa = log_nfa_new;
          rect_copy(&r,rec);
        }
    }

  return log_nfa;
}

/*----------------------------------------------------------------------------*/
/** Reduce the region size, by elimination the point2is far from the
    starting point2i, until that leads to rectangle with the right
    density of region point2is or to discard the region if too small.
 */
static int reduce_region_radius( struct point2i * reg, int * reg_size,
                                 image_double modgrad, double reg_angle,
                                 double prec, double p, struct rect * rec,
                                 image_char used, image_double angles,
                                 double density_th )
{
  double density,rad1,rad2,rad,xc,yc;
  int i;

  /* check parameters */
  if( reg == NULL ) error("reduce_region_radius: invalid point2ier 'reg'.");
  if( reg_size == NULL )
    error("reduce_region_radius: invalid point2ier 'reg_size'.");
  if( prec < 0.0 ) error("reduce_region_radius: 'prec' must be positive.");
  if( rec == NULL ) error("reduce_region_radius: invalid point2ier 'rec'.");
  if( used == NULL || used->data == NULL )
    error("reduce_region_radius: invalid image 'used'.");
  if( angles == NULL || angles->data == NULL )
    error("reduce_region_radius: invalid image 'angles'.");

  /* compute region point2is density */ //该密度判断已经在函数外判断过，应该可以不用在判断了吧
  density = (double) *reg_size /
                         ( dist(rec->x1,rec->y1,rec->x2,rec->y2) * rec->width );

  // if the density criterion is satisfied there is nothing to do 
  if( density >= density_th ) return TRUE;
  

  /* compute region's radius */
  xc = (double) reg[0].x;
  yc = (double) reg[0].y;
  rad1 = dist( xc, yc, rec->x1, rec->y1 );
  rad2 = dist( xc, yc, rec->x2, rec->y2 );
  rad = rad1 > rad2 ? rad1 : rad2;

  /* while the density criterion is not satisfied, remove farther pixels */
  while( density < density_th )
    {
      rad *= 0.75; /* reduce region's radius to 75% of its value */

      /* remove point2is from the region and update 'used' map */
      for(i=0; i<*reg_size; i++)
        if( dist( xc, yc, (double) reg[i].x, (double) reg[i].y ) > rad )
          {
            /* point2i not kept, mark it as NOTUSED */
            used->data[ reg[i].x + reg[i].y * used->xsize ] = NOTUSED;
            /* remove point2i from the region */
            reg[i].x = reg[*reg_size-1].x; /* if i==*reg_size-1 copy itself */
            reg[i].y = reg[*reg_size-1].y;
            --(*reg_size);
            --i; /* to avoid skipping one point2i */
          }

      /* reject if the region is too small.
         2 is the minimal region size for 'region2rect' to work. */
      if( *reg_size < 2 ) return FALSE;

      /* re-compute rectangle */
      region2rect(reg,*reg_size,modgrad,reg_angle,prec,p,rec);

      /* re-compute region point2is density */
      density = (double) *reg_size /
                         ( dist(rec->x1,rec->y1,rec->x2,rec->y2) * rec->width );
    }

  /* if this point2i is reached, the density criterion is satisfied */
  return TRUE;
}

/*----------------------------------------------------------------------------*/
/** Refine a rectangle.

    For that, an estimation of the angle tolerance is performed by the
    standard deviation of the angle at point2is near the region's
    starting point2i. Then, a new region is grown starting from the same
    point2i, but using the estimated angle tolerance. If this fails to
    produce a rectangle with the right density of region point2is,
    'reduce_region_radius' is called to try to satisfy this condition.
 */
static int refine( struct point2i * reg, int * reg_size, image_double modgrad,
                   double reg_angle, double prec, double p, struct rect * rec,
                   image_char used, image_double angles, double density_th )
{
  double angle,ang_d,mean_angle,tau,density,xc,yc,ang_c,sum,s_sum;
  int i,n;

  /* check parameters */
  if( reg == NULL ) error("refine: invalid point2ier 'reg'.");
  if( reg_size == NULL ) error("refine: invalid point2ier 'reg_size'.");
  if( prec < 0.0 ) error("refine: 'prec' must be positive.");
  if( rec == NULL ) error("refine: invalid point2ier 'rec'.");
  if( used == NULL || used->data == NULL )
    error("refine: invalid image 'used'.");
  if( angles == NULL || angles->data == NULL )
    error("refine: invalid image 'angles'.");

  /* compute region point2is density */
  density = (double) *reg_size /
                         ( dist(rec->x1,rec->y1,rec->x2,rec->y2) * rec->width );

  /* if the density criterion is satisfied there is nothing to do */
  if( density >= density_th ) return TRUE;

  /*------ First try: reduce angle tolerance ------*/

  /* compute the new mean angle and tolerance */
  xc = (double) reg[0].x;
  yc = (double) reg[0].y;
  ang_c = angles->data[ reg[0].x + reg[0].y * angles->xsize ];
  sum = s_sum = 0.0;
  n = 0;
  for(i=0; i<*reg_size; i++)
    {
      used->data[ reg[i].x + reg[i].y * used->xsize ] = NOTUSED;
      if( dist( xc, yc, (double) reg[i].x, (double) reg[i].y ) < rec->width )
        {
          angle = angles->data[ reg[i].x + reg[i].y * angles->xsize ];
          ang_d = angle_diff_signed(angle,ang_c);
          sum += ang_d;//加上角度差
          s_sum += ang_d * ang_d;//加上角度差的平方
          ++n;
        }
    }
  mean_angle = sum / (double) n;
  //以2倍标准差作为新的角度容忍度，最开始为22.5°*pi/180
  tau = 2.0 * sqrt( (s_sum - 2.0 * mean_angle * sum) / (double) n  +  mean_angle*mean_angle ); /* 2 * standard deviation */
  //以新的角度容忍度重新进行区域生长
  /* find a new region from the same starting point2i and new angle tolerance */
  region_grow(reg[0].x,reg[0].y,angles,reg,reg_size,&reg_angle,used,tau);

  /* if the region is too small, reject */
  if( *reg_size < 2 ) return FALSE;

  /* re-compute rectangle */
  region2rect(reg,*reg_size,modgrad,reg_angle,prec,p,rec);

  /* re-compute region point2is density */
  density = (double) *reg_size /
                      ( dist(rec->x1,rec->y1,rec->x2,rec->y2) * rec->width );

  /*------ Second try: reduce region radius ------*/
  if( density < density_th )
    return reduce_region_radius( reg, reg_size, modgrad, reg_angle, prec, p,
                                 rec, used, angles, density_th );

  /* if this point2i is reached, the density criterion is satisfied */
  return TRUE;
}
//--------------------------------------------------------
//my code
bool isArcSegment(point2i * reg, int reg_size, struct rect * main_rect, image_double modgrad,image_char used,image_char pol,
                         double prec, double p, rect * rect_up, rect * rect_down)
{
	//double dx,dy; //main_rect 's main angle vector
	point2i * reg_up = (point2i*)malloc(reg_size*sizeof(point2i));
	point2i * reg_down = (point2i*)malloc(reg_size*sizeof(point2i));
	int   reg_up_size,reg_down_size;
	double reg_up_theta,reg_down_theta;
	double reg_up_x,reg_up_y,reg_down_x,reg_down_y;
	double weight,sum;
	double temp1,temp2;
	int same_pol_cnt,opp_pol_cnt;
	int i;

	same_pol_cnt = opp_pol_cnt = 0;
	reg_up_size = reg_down_size = 0;

	for ( i = 0; i < reg_size; i++)
	{
		switch(pol->data[reg[i].y*pol->xsize+reg[i].x])
		{
			case SAME_POL: same_pol_cnt++;break;//统计同极性的pixel数量
			case OPP_POL : opp_pol_cnt++; break;//统计反极性的pixel数量
			default:break;
		}
	 //选与theta角度为法线方向，过质心的直线方程为 dx*(x-xi)+dy*(y-yi)=0,则与方向相同的点代入方程得到距离d,d>=0归入reg_up,d<0归入reg_down
	  if( main_rect->dx*( reg[i].x - main_rect->x ) + main_rect->dy*( reg[i].y - main_rect->y ) >= 0)
		  reg_up[reg_up_size++] = reg[i];
	  else
		  reg_down[reg_down_size++] = reg[i];
	}
	//对于已经被标记过极性的区域，我们没必要再进行极性分析
	if( (same_pol_cnt + opp_pol_cnt) > reg_size/2)
	{
		if(same_pol_cnt > opp_pol_cnt )
		{
			main_rect->polarity = 1;
		    rect_up->polarity = 1;
	        rect_down->polarity = 1;
		}
		else
		{
			main_rect->polarity = -1;
		    rect_up->polarity = -1;
	        rect_down->polarity = -1;
		}
		return TRUE;
	}
	//计算与主方向相同的上半部分区域质心
	reg_up_x = reg_up_y = sum = 0;
	for ( i = 0; i< reg_up_size; i++)
	{
		weight = modgrad->data[ reg_up[i].x + reg_up[i].y * modgrad->xsize ];
		reg_up_x += (double)weight*reg_up[i].x;
		reg_up_y += (double)weight*reg_up[i].y;
		sum += weight;
	}
	reg_up_x /= sum;
	reg_up_y /= sum;
	//计算主方向上的下半部分区域质心
	reg_down_x = reg_down_y = sum = 0;
	for ( i = 0; i< reg_down_size; i++)
	{
		weight = modgrad->data[ reg_down[i].x + reg_down[i].y * modgrad->xsize ];
		reg_down_x += (double)weight*reg_down[i].x;
		reg_down_y += (double)weight*reg_down[i].y;
		sum += weight;
	}
	reg_down_x /= sum;
	reg_down_y /= sum;
	//估计两个区域方向
	reg_up_theta = get_theta(reg_up,reg_up_size,reg_up_x,reg_up_y,modgrad,main_rect->theta,prec);
	reg_down_theta = get_theta(reg_down,reg_down_size,reg_down_x,reg_down_y,modgrad,main_rect->theta,prec);
	//旋转到0°进行比较theta,reg_up_theta,reg_down_theta
	temp1 = angle_diff_signed(reg_up_theta,main_rect->theta);
	temp2 = angle_diff_signed(reg_down_theta,main_rect->theta);
	/*if(temp1>= M_PI/2 || temp1 <= -M_PI/2)
		temp1 += 0;
	if(temp2>= M_PI/2 || temp2 <= -M_PI/2)
		temp2 += 0;*/
	//if(temp1 >= prec/10 && temp2 <= -prec/10)//顺时针,边缘的梯度方向与弧的指向圆心方向相反，polarity = -1
	if(temp1 >= M_1_8_PI/10 && temp2 <= -M_1_8_PI/10)//实验证明取定值效果更好
	{
		main_rect->polarity = -1;
		rect_up->polarity = -1;
	    rect_down->polarity = -1;
		//标记极性
	    for ( i = 0; i < reg_size; i++)
	    {
			pol->data[reg[i].y*pol->xsize+reg[i].x] = OPP_POL;//-1
	    }
	}
	//else if(temp1 <= -prec/10 && temp2 >= prec/10)//逆时针，边缘的梯度方向与弧的指向圆心方向相同，polarity = 1
	else if(temp1 <= -M_1_8_PI/10 && temp2 >= M_1_8_PI/10)//实验证明取定值效果更好
	{
		main_rect->polarity = 1;
		rect_up->polarity = 1;
	    rect_down->polarity = 1;
		//标记极性
	    for ( i = 0; i < reg_size; i++)
	    {
			pol->data[reg[i].y*pol->xsize+reg[i].x] = SAME_POL;//1
	    }
	}
	else
	{
		//在region_grow中已经置为USED了
		//for ( i = 0; i< reg_size; i++)
		//	used->data[reg[i].y*used->xsize+reg[i].x] = USED;
		return FALSE;
	}
	
	//region2rect2(reg_up,reg_up_size,reg_up_x,reg_up_y,reg_up_theta,prec,p,rect_up);
	//region2rect2(reg_down,reg_down_size,reg_down_x,reg_down_y,reg_down_theta,prec,p,rect_down);

	free(reg_up);
	free(reg_down);
	return TRUE;
}

/*----------------------------------------------------------------------------*/
/*-------------------------- Line Segment Detector ---------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** LSD full interface.
 */
double * LineSegmentDetection( int * n_out,
                               double * img, int X, int Y,
                               double scale, double sigma_scale, double quant,
                               double ang_th, double log_eps, double density_th,
                               int n_bins,
                               int ** reg_img, int * reg_x, int * reg_y )
{
  image_double image;
  ntuple_list out = new_ntuple_list(8);
  double * return_value;
  image_double scaled_image,angles,modgrad;
  image_char used;
  image_char pol;  //对于构成圆弧的像素标记极性，如果梯度的方向和弧的方向指向一致，则为SAME_POLE,否则为OPP_POLE,该标记初始是为0
  image_int region = NULL;
  struct coorlist * list_p;
  struct coorlist * list_p_temp;
//  struct coorlist * mem_p;
  struct rect main_rect;//main rect
  struct rect rect_up,rect_down;//divide the rect into 2 rects:rect_up and rect_down
  struct point2i * reg;
  int reg_size,min_reg_size,i;
  unsigned int xsize,ysize;
  double rho,reg_angle,prec,p;
  double log_nfa = -1,logNT;
//  double log_nfa1,log_nfa2;
  int ls_count = 0;                   /* line segments are numbered 1,2,3,... */
  int seed_cnt = 0;
  int refine_cnt = 0;
  int reg_size_toosmall_cnt=0;

  /* check parameters */
  if( img == NULL || X <= 0 || Y <= 0 ) error("invalid image input.");
  if( scale <= 0.0 ) error("'scale' value must be positive.");
  if( sigma_scale <= 0.0 ) error("'sigma_scale' value must be positive.");
  if( quant < 0.0 ) error("'quant' value must be positive.");
  if( ang_th <= 0.0 || ang_th >= 180.0 )
    error("'ang_th' value must be in the range (0,180).");
  if( density_th < 0.0 || density_th > 1.0 )
    error("'density_th' value must be in the range [0,1].");
  if( n_bins <= 0 ) error("'n_bins' value must be positive.");


  /* angle tolerance */
  prec = M_PI * ang_th / 180.0;
  p = ang_th / 180.0;

  rho = quant / sin(prec); /* gradient magnitude threshold */


  /* load and scale image (if necessary) and compute angle at each pixel */
  image = new_image_double_ptr( (unsigned int) X, (unsigned int) Y, img );
  if( scale != 1.0 )
    {
	  //按照scale进行高斯降采样的图像，注意宽高是上取整，设采样后高宽为imgN*imgM
      scaled_image = gaussian_sampler( image, scale, sigma_scale );
	  //返回一张梯度角度顺时针旋转90°后的align角度图angles，如果梯度角度是(gx,gy)->(-gy,gx)，
	  //和梯度的模的图modgrad,然后按照n_bins进行伪排序返回链表的头指针list_p,里面存的是坐标
	  angles = ll_angle( scaled_image, rho, &list_p,&modgrad, (unsigned int) n_bins );
      free_image_double(scaled_image);
    }
  else
    angles = ll_angle( image, rho, &list_p,&modgrad,(unsigned int) n_bins );
  xsize = angles->xsize;//降采样后的图像的x size，宽度imgM
  ysize = angles->ysize;//降采样后的图像的y size，高度imgN

  /* Number of Tests - NT

     The theoretical number of tests is Np.(XY)^(5/2)
     where X and Y are number of columns and rows of the image.
     Np corresponds to the number of angle precisions considered.
     As the procedure 'rect_improve' tests 5 times to halve the
     angle precision, and 5 more times after improving other factors,
     11 different precision values are potentially tested. Thus,
     the number of tests is
       11 * (X*Y)^(5/2)
     whose logarithm value is
       log10(11) + 5/2 * (log10(X) + log10(Y)).
  */
  logNT = 5.0 * ( log10( (double) xsize ) + log10( (double) ysize ) ) / 2.0
          + log10(11.0);
  min_reg_size = (int) (-logNT/log10(p)); /* minimal number of point2is in region that can give a meaningful event，每个矩形区域内align point2i最小数量*/
  /* initialize some structures */
  if( reg_img != NULL && reg_x != NULL && reg_y != NULL ) /* save region data */
    region = new_image_int_ini(angles->xsize,angles->ysize,0);//申请与降采样后图像一样大小的int类型的内存，该内存的作用是将检测到的线段序号标到相应的图像格子里，该部分可有可无
  used = new_image_char_ini(xsize,ysize,NOTUSED);//申请与降采样后图像一样大小的char类型的内存
  pol  = new_image_char_ini(xsize,ysize,NOTDEF_POL);//像素点处的梯度和弧指向的方向的极性标记
  reg = (struct point2i *) calloc( (size_t) (xsize*ysize), sizeof(struct point2i) );
  if( reg == NULL ) error("not enough memory!");

  list_p_temp = list_p;//记录头链表的头指针，后面需要利用该头指针进行内存释放
  /* search for line segments */
  for(; list_p_temp != NULL; list_p_temp = list_p_temp->next )
    if( used->data[ list_p_temp->x + list_p_temp->y * used->xsize ] == NOTUSED &&
        angles->data[ list_p_temp->x + list_p_temp->y * angles->xsize ] != NOTDEF )
       /* there is no risk of double comparison problems here
          because we are only interested in the exact NOTDEF value */
      {
        /* find the region of connected point2i and ~equal angle */
		//reg是长度为imgN*imgM的一维point2i型数组，有足够大的空间存储生长的区域，reg_size是里面存储了数据的数量，记录的是区域的point2i
		//reg_angle是该区域的主方向的double型变量，存的角度是弧度制
		  seed_cnt ++;
        region_grow( list_p_temp->x, list_p_temp->y, angles, reg, &reg_size,&reg_angle, used, prec );

        /* reject small regions */
        if( reg_size < min_reg_size ) 
		{
			reg_size_toosmall_cnt++;
			continue;
		}

        /* construct rectangular approximation for the region */
		//根据生长的区域得到近似外接矩阵的参数，矩形参数包括:起点，终点，方向theta，宽度等
        region2rect(reg,reg_size,modgrad,reg_angle,prec,p,&main_rect);
		if( FALSE == isArcSegment(reg,reg_size,&main_rect,modgrad,used,pol,prec,p,&rect_up,&rect_down))
			continue;
        /* Check if the rectangle exceeds the minimal density of
           region point2is. If not, try to improve the region.
           The rectangle will be rejected if the final one does
           not fulfill the minimal density condition.
           This is an addition to the original LSD algorithm published in
           "LSD: A Fast Line Segment Detector with a False Detection Control"
           by R. Grompone von Gioi, J. Jakubowicz, J.M. Morel, and G. Randall.
           The original algorithm is obtained with density_th = 0.0.
         */

        //提纯，通过重新生长区域来达到期望的密度阈值 
        if( !refine( reg, &reg_size, modgrad, reg_angle,
                     prec, p, &main_rect, used, angles, density_th ) ) continue;

		refine_cnt++;
        // compute NFA value 
        log_nfa = rect_improve(&main_rect,angles,logNT,log_eps);//通过改善矩形区域以尝试得到期望的nfa值
        if( log_nfa <= log_eps ) //错误控制
			continue;
        // A New Line Segment was found! 
        ++ls_count;  // increase line segment counter 

        //
        //  The gradient was computed with a 2x2 mask, its value corresponds to
        //  point2is with an offset of (0.5,0.5), that should be added to output.
        //  The coordinates origin is at the center of pixel (0,0).
        //
        main_rect.x1 += 0.5; main_rect.y1 += 0.5;
        main_rect.x2 += 0.5; main_rect.y2 += 0.5;

        // scale the result values if a subsampling was performed */
        if( scale != 1.0 )
          {
            main_rect.x1 /= scale; main_rect.y1 /= scale;
            main_rect.x2 /= scale; main_rect.y2 /= scale;
            main_rect.width /= scale;
          }

        /* add line segment found to output */
		add_8tuple( out, main_rect.x1, main_rect.y1, main_rect.x2, main_rect.y2,main_rect.dx,main_rect.dy,
			        main_rect.width, main_rect.polarity);

		//------------------------------------------------------------------------------------------------- 
		/*
		cout<<ls_count<<'\t'<<main_rect.theta<<'\t'<<main_rect.theta*180/M_PI<<"\t polarity:"<<main_rect.polarity<<endl;//打印theta
		
			fstream file1,file2;
			if(ls_count == 1)//清空内容
			{
				file1.open("D:\\Graduate Design\\picture\\sp\\coor.txt",ios::out | ios::trunc);
				file1.close();
				file2.open("D:\\Graduate Design\\picture\\sp\\reg.txt",ios::out | ios::trunc);
				file2.close();
			}
			
			file1.open("D:\\Graduate Design\\picture\\sp\\coor.txt",ios::app);
			file1<<main_rect.x1<<'\t'<<main_rect.y1<<'\t'<<main_rect.x2<<'\t'<<main_rect.y2<<'\t'<<(main_rect.theta*180/M_PI)<<endl;
			file1.close();
			
			if(ls_count == 1)//保持第1根线段的区域
			{
				file2.open("D:\\Graduate Design\\picture\\sp\\reg.txt",ios::app);
				for(i=0; i<reg_size; i++)
					file2<<angles->data[ reg[i].x + reg[i].y * angles->xsize ]*180/M_PI<<endl;
				file2.close();
			}
			*/
		//-------------------------------------------------------------------------------------------------------
        /* add region number to 'region' image if needed */ //将检测到的线段序号标到相应的图像格子里，该部分可有可无
        if( region != NULL )
          for(i=0; i<reg_size; i++)
            region->data[ reg[i].x + reg[i].y * region->xsize ] = ls_count;
      }


  /* free memory */
  free( (void *) image );   /* only the double_image structure should be freed,
                               the data point2ier was provided to this functions
                               and should not be destroyed.                 */
  free_image_double(angles);
  free_image_double(modgrad);
  free_image_char(used);
  free_image_char(pol);
  free( (void *) reg );
//  free( (void *) mem_p );
  //释放分成1024区的存储梯度从大到小的链表,mycode
  //---------------------------------------
  list_p_temp = list_p->next;
  while(list_p_temp != NULL)
  {
	  free(list_p);
	  list_p = list_p_temp;
	  list_p_temp = list_p->next;
  }
  free(list_p);

  //cout<<"seed cnt:"<<seed_cnt<<endl;
  //cout<<"refine cnt:"<<refine_cnt<<endl;
  //cout<<"reg_size_toosmall cnt:"<<reg_size_toosmall_cnt<<endl;
  //----------------------------------------
  /* return the result */
  if( reg_img != NULL && reg_x != NULL && reg_y != NULL )
    {
      if( region == NULL ) error("'region' should be a valid image.");
      *reg_img = region->data;
      if( region->xsize > (unsigned int) INT_MAX ||
          region->xsize > (unsigned int) INT_MAX )
        error("region image to big to fit in INT sizes.");
      *reg_x = (int) (region->xsize);
      *reg_y = (int) (region->ysize);

      /* free the 'region' structure.
         we cannot use the function 'free_image_int' because we need to keep
         the memory with the image data to be returned by this function. */
      free( (void *) region );
    }
  if( out->size > (unsigned int) INT_MAX )
    error("too many detections to fit in an INT.");
  *n_out = (int) (out->size);

  return_value = out->values;
  free( (void *) out );  /* only the 'ntuple_list' structure must be freed,
                            but the 'values' point2ier must be keep to return
                            as a result. */
  return return_value;
}

/*------------------------------------------------------------------------------------------------*/
/**
my code,Alan Lu
输入
img  : 输入图像的一维double型数组,大小为Y*X，按照行优先存储，传入前需要拥有内存
X    : 输入图像的columns
Y    ：输入图像的rows
输出
n_out: lsd算法检测得到的线段的数量n，return的返回值是n条线段，为一维double型数组，长度为8*n，每8个为一组，存着x1,y1,x2,y2,dx,dy,width,polarity
reg_img: 输出标记区域，是一维的int型数组，大小reg_y*reg_x,在相应的像素位置标记着它属于的线段(1,2,3,...n),如果值为0表示不属于任何线段.
         假如外部是int * region_img,则只需要 &region_img,就可以得到标记区域的返回，不需要时直接NULL传入
reg_x  : 输出标记区域的columns,不需要时直接NULL传入
reg_y  : 输出标记区域的rows,不需要时直接NULL传入
*/
double * mylsd(int * n_out, double * img, int X, int Y, int ** reg_img, int * reg_x, int * reg_y)
{
	 /* LSD parameters */
  double scale = 0.8;       /* Scale the image by Gaussian filter to 'scale'. */
  double sigma_scale = 0.6; /* Sigma for Gaussian filter is computed as
                                sigma = sigma_scale/scale.                    */
  double quant = 2.0;       /* Bound to the quantization error on the
                                gradient norm.                                */
  double ang_th = 45;     /* Gradient angle tolerance in degrees.           */
  double log_eps = 0.0;     /* Detection threshold: -log10(NFA) > log_eps     */
  double density_th = 0.7;  /* Minimal density of region point2is in rectangle. */
  int n_bins = 1024;        /* Number of bins in pseudo-ordering of gradient
                               modulus.                                       */ 

  return LineSegmentDetection( n_out, img, X, Y, scale, sigma_scale, quant,
                               ang_th, log_eps, density_th, n_bins,
                               reg_img, reg_x, reg_y );
}
//===================================================================================================
//================================Generate Cirlce Candidates=========================================
//匹配线段对，线段对的索引参数，圆参数
typedef struct PairedSegment_s
{
	point2i pairedSegInd;
	point3d circle;
}PairedSegment;

//匹配线段对结点
typedef struct PairedSegmentNode_s
{
	point2i pairedSegInd;
	point3d circle;
	PairedSegmentNode_s* next;
}PairedSegmentNode;

typedef struct  PairedSegmentList_s
{
	int length;
	PairedSegment * pairSeg;
}PairedSegmentList;

typedef struct Point2dNode_s
{
	point2d point;
	Point2dNode_s * next;
}Point2dNode;

typedef struct Point3dNode_s
{
	point3d point;
	Point3dNode_s * next;
}Point3dNode;

typedef struct Point1dNode_s
{
	double data;
	Point1dNode_s * next;
}Point1dNode;
PairedSegmentList * pairedSegmentListInit( int length)
{
	if(length <= 0)
		error("paired segment length less equal than 0");
	PairedSegmentList * pairedSegList = (PairedSegmentList*)malloc(sizeof(PairedSegmentList));
	pairedSegList->length = length;
	pairedSegList->pairSeg = (PairedSegment*)malloc(sizeof(PairedSegment)*length);
	if(pairedSegList->pairSeg == NULL)
		error("pairedSegmentListInit,not enough memory");
	return pairedSegList;
}

void freePairedSegmentList( PairedSegmentList * list)
{
	if(list == NULL || list->pairSeg == NULL)
		error("freePairedSegmentList,invalidate free");
	free(list->pairSeg);
	free(list);
	list = NULL;
}

//计算梯度，返回模和角度，同时模值太小的像素点直接抑制掉，赋值为NOTDEF
//mod、angles为了传值，是二级指针
void calculateGradient( double * img_in, unsigned int imgx, unsigned int imgy,image_double * mod, image_double * angles)
{
	if(img_in == NULL || imgx == 0 || imgy == 0)
		error("calculateGradient error!");
	(*mod) = new_image_double(imgx,imgy);
	(*angles) = new_image_double(imgx,imgy);
	double threshold = 2/sin(22.5/180*M_PI);
	unsigned int x,y,adr;
	double com1,com2;
	double gx,gy;
	double norm,norm_square;
	double sum = 0;

	//double max_grad = 0.0;
	//边界初始为NOTDEF
	for ( x = 0; x<imgx; x++) 
	{
		//(*angles)->data[x]=NOTDEF;
		(*angles)->data[(imgy-1)*imgx+x]=NOTDEF;
		//(*mod)->data[x]=NOTDEF;
		(*mod)->data[(imgy-1)*imgx+x]=NOTDEF;
	}
	for ( y = 0; y<imgy; y++) 
	{
		//(*angles)->data[y*imgx] = NOTDEF;
		(*angles)->data[y*imgx+imgx-1] = NOTDEF;
		//(*mod)->data[y*imgx] = NOTDEF;
		(*mod)->data[y*imgx+imgx-1] = NOTDEF;
	}
	 /* compute gradient on the remaining pixels */
	for(x=0;x<imgx-1;x++)
		for(y=0;y<imgy-1;y++)
		{
			adr = y*imgx+x;
		  /*
		     Norm 2 computation using 2x2 pixel window:
		       A B
		       C D
		     and
		       com1 = D-A,  com2 = B-C.
		     Then
		       gx = B+D - (A+C)   horizontal difference
		       gy = C+D - (A+B)   vertical difference
		     com1 and com2 are just to avoid 2 additions.
		   */
		  com1 = img_in[adr+imgx+1] - img_in[adr];
		  com2 = img_in[adr+1]   - img_in[adr+imgx];

		  gx = com1+com2; /* gradient x component */
		  gy = com1-com2; /* gradient y component */
		  norm_square = gx*gx+gy*gy;

		  norm = sqrt( norm_square / 4.0 ); /* gradient norm */

		  (*mod)->data[adr] = norm; /* store gradient norm */

		  if( norm <= threshold ) /* norm too small, gradient no defined */
		  {
		    (*angles)->data[adr] = NOTDEF; /* gradient angle not defined */
			(*mod)->data[adr] = NOTDEF;
		  }
		  else
		    {
		      /* gradient angle computation */
		      (*angles)->data[adr] = atan2(gx,-gy);
		    }
		}
}

void calculateGradient2( double * img_in, unsigned int imgx, unsigned int imgy, image_double * angles)
{
	if(img_in == NULL || imgx == 0 || imgy == 0)
		error("calculateGradient error!");
	image_double mod = new_image_double(imgx,imgy);
	(*angles) = new_image_double(imgx,imgy);
	unsigned int x,y,adr;
	double com1,com2;
	double gx,gy;
	double norm,norm_square;
	double threshold;
	double sum = 0;
	double value;  
	//double max_grad = 0.0;
	//边界初始为NOTDEF
	for ( x = 0; x<imgx; x++) 
	{
		(*angles)->data[x]=NOTDEF;
		(*angles)->data[(imgy-1)*imgx+x]=NOTDEF;
		(mod)->data[x]=NOTDEF;
		(mod)->data[(imgy-1)*imgx+x]=NOTDEF;
	}
	for ( y = 0; y<imgy; y++) 
	{
		(*angles)->data[y*imgx] = NOTDEF;
		(*angles)->data[y*imgx+imgx-1] = NOTDEF;
		(mod)->data[y*imgx] = NOTDEF;
		(mod)->data[y*imgx+imgx-1] = NOTDEF;
	}
	 /* compute gradient on the remaining pixels */
	for(x=1;x<imgx-1;x++)
		for(y=1;y<imgy-1;y++)
		{
			adr = y*imgx+x;
		  /*
		     Norm 2 computation using 2x2 pixel window:
		       A B C
		       D E F
			   G H I
		     and
		       com1 = C-G,  com2 = I-A.
		     Then
		       gx = C+2F+I - (A+2D+G)=com1+com2+2(F-D)   horizontal derivative
		       gy = G+2H+I - (A+2B+C)=-com1+com2+2(H-B)   vertical derivative
		     com1 and com2 are just to avoid 2 additions.
		   */
		  com1 = img_in[adr-imgx+1] - img_in[adr+imgx-1];
		  com2 = img_in[adr+imgx+1] - img_in[adr-imgx-1];

		  gx = (com1+com2+2*(img_in[adr+1] - img_in[adr-1]))/(8.0*255); /* gradient x component */
		  gy = (-com1+com2+2*(img_in[adr+imgx] - img_in[adr-imgx]))/(8.0*255); /* gradient y component */
		  norm_square = gx*gx+gy*gy;
		  sum+=norm_square;

		  norm = sqrt( norm_square); /* gradient norm */

		  (mod)->data[adr] = norm; /* store gradient norm */
		   /* gradient angle computation */
	     (*angles)->data[adr] = atan2(gy,gx);
		}
	threshold = 2*sqrt(sum/(imgx*imgy));//自动阈值
	//non maximum suppression
	for(x=1;x<imgx-1;x++)
		for(y=1;y<imgy-1;y++)
		{
			adr = y*imgx+x;
			value = (*angles)->data[adr];
			if((mod)->data[adr] < threshold )
			{
				(*angles)->data[adr] = NOTDEF;
				continue;
			}
			if( (value > -M_1_8_PI && value<=M_1_8_PI) || (value <= -M_7_8_PI ) || (value > M_7_8_PI))
			{
				if((mod)->data[adr] <= (mod)->data[adr+1] || (mod)->data[adr] <= (mod)->data[adr-1])
					(*angles)->data[adr] = NOTDEF;
			}
			else if( (value> M_1_8_PI && value<= M_3_8_PI) || (value> -M_7_8_PI && value<= -M_5_8_PI) )
			{
				if((mod)->data[adr] <= (mod)->data[adr-imgx-1] || (mod)->data[adr] <= (mod)->data[adr+imgx+1])
					(*angles)->data[adr] = NOTDEF;
			}
			else if((value> M_3_8_PI && value<= M_5_8_PI) || (value> -M_5_8_PI && value<= -M_3_8_PI))
			{
				if((mod)->data[adr] <= (mod)->data[adr-imgx] || (mod)->data[adr] <= (mod)->data[adr+imgx])
					(*angles)->data[adr] = NOTDEF;
			}
			else 
			{
				if((mod)->data[adr] <= (mod)->data[adr-imgx+1] || (mod)->data[adr] <= (mod)->data[adr+imgx-1])
					(*angles)->data[adr] = NOTDEF;
			}
		}
    //也标记到mod图上面
	//for(x=1;x<imgx-1;x++)
	//	for(y=1;y<imgy-1;y++)
	//	{
	//		if((*angles)->data[y*imgx+x] == NOTDEF)
	//			(mod)->data[y*imgx+x] = NOTDEF;
	//	}
		free_image_double(mod);
}

inline bool linearProgram( point2d a, point2d b, point2d line1_dir, point2d c, point2d d, point2d line2_dir,  double polarity, double linear_prog_dis_tolerance)
{
	point2d arc1_vec,arc2_vec,test_vec1,test_vec2,test_vec3; //弧指向圆心的向量和测试向量
	if( polarity == 1)// polarity is equal 1, arc vector = (dy,-dx)
	{
		arc1_vec.x = line1_dir.y;
		arc1_vec.y = -line1_dir.x;
		arc2_vec.x = line2_dir.y;
		arc2_vec.y = -line2_dir.x;
	}
	else// polarity is equal -1, arc vector = (-dy,dx)
	{
		arc1_vec.x = -line1_dir.y;
		arc1_vec.y = line1_dir.x;
		arc2_vec.x = -line2_dir.y;
		arc2_vec.y = line2_dir.x;
	}
	/*
	A-------B     line i
	C-------D     line j
	test_vec1 = AC
	test_vec2 = AD
	test_vec3 = BC
	*/
	//method 1
	test_vec1.x = c.x - a.x;
	test_vec1.y = c.y - a.y;
	test_vec2.x = d.x - a.x;
	test_vec2.y = d.y - a.y;
	test_vec3.x = c.x - b.x;
	test_vec3.y = c.y - b.y;
	if( dotProduct(arc1_vec,test_vec1) >= linear_prog_dis_tolerance && 
		dotProduct(arc1_vec,test_vec2) >= linear_prog_dis_tolerance &&  
		-dotProduct(arc2_vec,test_vec1) >= linear_prog_dis_tolerance &&  
		-dotProduct(arc2_vec,test_vec1) >= linear_prog_dis_tolerance 
		) //space linear program
		return TRUE;
	//method 2
	/*test_vec1.x = (c.x + d.x - a.x - b.x)/2;
	test_vec1.y = (c.y + d.y - a.y - b.y)/2;
	test_vec2.x = - test_vec1.x;
	test_vec2.y = - test_vec1.y;
	if( dotProduct(arc1_vec,test_vec1) >= linear_prog_dis_tolerance &&
		dotProduct(arc2_vec,test_vec2) >= linear_prog_dis_tolerance )
		return TRUE;*/
	return FALSE;
}
inline bool calcCircleParametersAndValidate( double * lines, int line_num, int first_line_ind,int second_line_ind, image_double angles, double distance_tolerance, point3d * circle )
{
	/*
	dx1*(x-x1)+dy1*(y-y1)=0
	dx2*(x-x2)+dy2*(y-y2)=0
	x  =  c1*dy2-c2*dy1   /
	y    -c1*dx2+c2*dx1  /  (dx1*dy2 - dx2*dy1)

	AX = C => X = A^{-1}C
	*/
	point2d equationC;
	double  delta;
	point2d segCenter1,segCenter2;
	point2d line1_dir,line2_dir;
	double r1,r2;
	rect rec1,rec2;
	int validate_cnt,total_cnt;
	if(first_line_ind >= line_num  || second_line_ind >= line_num )
		error("calcCircleParametersAndValidate, line index corrupt");
	//calculate the circle (x,y,r)
	rec1.x1 = lines[first_line_ind*8];
	rec1.y1 = lines[first_line_ind*8+1];
	rec1.x2 = lines[first_line_ind*8+2];
	rec1.y2 = lines[first_line_ind*8+3];
	rec1.x  = (rec1.x1 + rec1.x2)/2;
	rec1.y  = (rec1.y1 + rec1.y2)/2;
	rec1.dx = lines[first_line_ind*8+4];
	rec1.dy = lines[first_line_ind*8+5];
	rec1.width = 2*distance_tolerance;  //精度更高
	rec1.polarity = (int)lines[first_line_ind*8+7];
	rec2.x1 = lines[second_line_ind*8];
	rec2.y1 = lines[second_line_ind*8+1];
	rec2.x2 = lines[second_line_ind*8+2];
	rec2.y2 = lines[second_line_ind*8+3];
	rec2.x  = (rec2.x1 + rec2.x2)/2;
	rec2.y  = (rec2.y1 + rec2.y2)/2;
	rec2.dx = lines[second_line_ind*8+4];
	rec2.dy = lines[second_line_ind*8+5];
	rec2.width = 2*distance_tolerance;  //精度更高
	segCenter1.x = rec1.x;
	segCenter1.y = rec1.y;
	segCenter2.x = rec2.x;
	segCenter2.y = rec2.y;
	line1_dir.x  = rec1.dx;
	line1_dir.y  = rec1.dy;
	line2_dir.x  = rec2.dx;
	line2_dir.y  = rec2.dy;
	equationC.x = dotProduct(segCenter1,line1_dir);
	equationC.y = dotProduct(segCenter2,line2_dir);
	delta = line1_dir.x*line2_dir.y - line1_dir.y*line2_dir.x;//由于事先只让夹角>=15°的线段对参与计算交点，因此判别式不会等于0.
	circle->x = ( equationC.x * line2_dir.y - equationC.y*line1_dir.y)/delta;
	circle->y = (-equationC.x * line2_dir.x + equationC.y*line1_dir.x)/delta;
	r1  = dist(circle->x, circle->y,rec1.x1,rec1.y1);//半径
    r2  = dist(circle->x, circle->y,rec2.x1,rec2.y1);
	//validate the radius
	if( abs(r1-r2) <= 2*distance_tolerance && min(r1,r2) > 3*distance_tolerance && max(r1,r2) < min(angles->xsize,angles->ysize) &&  circle->x > 0 && circle->x < angles->xsize && circle->y > 0 && circle->y < angles->ysize )
	{
		circle->r = (r1+r2)/2;
		//validate the distribution rates of inliers
		rect_iter * ri1,*ri2;
		double point_normal,temp;
		if(rec1.polarity == 1)//极性一致，梯度方向和圆弧指向方向相同，因此梯度方向和圆弧支持像素点指向圆心的法线方向向量相同
		{
			validate_cnt = total_cnt = 0;
			for(ri1 = ri_ini(&rec1);!ri_end(ri1);ri_inc(ri1))
			{
				//外接矩形可能会越界
				if(ri1->x >= 0 && ri1->y >= 0 && ri1->x < angles->xsize && ri1->y < angles->ysize)
				{
					temp  = angles->data[ri1->y*angles->xsize+ri1->x] ;//内点的梯度方向
					if(temp!= NOTDEF )
					{
						point_normal = atan2(circle->y - ri1->y,circle->x - ri1->x); //边缘点的法线方向
						total_cnt++;
						if(angle_diff(point_normal,temp) <= M_1_9_PI && abs(dist(ri1->x,ri1->y,circle->x,circle->y)-circle->r) <= 3*distance_tolerance) //+- 20°内 且 || d - r || < 3 dis_t
							validate_cnt++;
					}
				}
			}
			if(validate_cnt > 0 && validate_cnt*1.0/total_cnt >= 0.6)
			{
				validate_cnt = total_cnt = 0;
				for(ri2 = ri_ini(&rec2);!ri_end(ri2);ri_inc(ri2))
				{
					//外接矩形可能会越界
					if(ri2->x >= 0 && ri2->y >= 0 && ri2->x < angles->xsize && ri2->y < angles->ysize)
					{
						temp  = angles->data[ri2->y*angles->xsize+ri2->x] ;//内点的梯度方向
						if(temp!= NOTDEF )
						{
							point_normal = atan2(circle->y - ri2->y,circle->x - ri2->x); //边缘点的指向圆心侧法线方向
							total_cnt++;
							if(angle_diff(point_normal,temp) <= M_1_9_PI && abs(dist(ri2->x,ri2->y,circle->x,circle->y)-circle->r) <= 3*distance_tolerance) //+- 20°内 且 || d - r || < 3 dis_t
								validate_cnt++;
						}
					}
				}
				if(validate_cnt > 0 && validate_cnt*1.0/total_cnt >= 0.6)
					return TRUE;
			}
		}
		else//极性相反，梯度方向和圆弧指向方向相反，因此梯度方向和圆弧支持像素点指向圆心的法线方向向量相反
		{
			validate_cnt = total_cnt = 0;
			for(ri1 = ri_ini(&rec1);!ri_end(ri1);ri_inc(ri1))
			{
				//外接矩形可能会越界
				if(ri1->x >= 0 && ri1->y >= 0 && ri1->x < angles->xsize && ri1->y < angles->ysize)
				{
					temp  = angles->data[ri1->y*angles->xsize+ri1->x] ;//内点的梯度方向
					if(temp!= NOTDEF )
					{
						point_normal = atan2(ri1->y-circle->y, ri1->x-circle->x);  //圆心指向边缘点侧法线方向
						total_cnt++;
						if(angle_diff(point_normal,temp) <= M_1_9_PI && abs(dist(ri1->x,ri1->y,circle->x,circle->y)-circle->r) <= 3*distance_tolerance) //+- 20°内 且 || d - r || < 3 dis_t
							validate_cnt++;
					}
				}
			}
			if(validate_cnt > 0 && validate_cnt*1.0/total_cnt >= 0.6)
			{
				validate_cnt = total_cnt = 0;
				for(ri2 = ri_ini(&rec2);!ri_end(ri2);ri_inc(ri2))
				{
					//外接矩形可能会越界
					if(ri2->x >= 0 && ri2->y >= 0 && ri2->x < angles->xsize && ri2->y < angles->ysize)
					{
						temp  = angles->data[ri2->y*angles->xsize+ri2->x] ;//内点的梯度方向
						if(temp!= NOTDEF )
						{
							point_normal = atan2( ri2->y-circle->y, ri2->x-circle->x); //边缘点的法线方向
							total_cnt++;
							if(angle_diff(point_normal,temp) <= M_1_9_PI && abs(dist(ri2->x,ri2->y,circle->x,circle->y)-circle->r) <= 3*distance_tolerance) //+- 20°内 且 || d - r || < 3 dis_t
								validate_cnt++;
						}
					}
				}
				if(validate_cnt > 0 && validate_cnt*1.0/total_cnt >= 0.6)
					return TRUE;
			}
		}
	}
	return FALSE;
}
//lsd算法检测得到的线段的数量line_nums，return的返回值是line_nums条线段，为一维double型数组lines，长度为8*n，每8个为一组
//存着x1,y1,x2,y2,dx,dy,width,polarity
PairedSegmentList * getValidatePairedSegment( double * lines, int line_num, image_double angles, double distance_tolerance)
{
	PairedSegmentList * pairSegmentList = NULL;
	PairedSegmentNode *head, *tail;
	int pairlength = 0;
	point2d pointA,pointB,pointC,pointD,ab_dir,cd_dir;
	point3d circle;
	double polarity;
	//double distance_tolerance = max( 2.0, 0.005*min(angles->xsize,angles->ysize) ); // 0.005%*min(xsize,ysize)
	int i,j;
    double temp_dot;
	//pairSegmentList = pairedSegmentListInit(0);
	head = tail = NULL;
	for ( i = 0; i<line_num-1; i++)
		for ( j = i+1; j<line_num; j++)
		{
			//line i 's polarity is the same as line j
			//intersection angle great than 0° and less than 180°
            temp_dot = (lines[i*8+4]*lines[j*8+4]+lines[i*8+5]*lines[j*8+5]);
			if( lines[i*8+7] == lines[j*8+7] &&  temp_dot < 1 && temp_dot> 0) //great than 90°
			{
				pointA.x = lines[i*8];
				pointA.y = lines[i*8+1];
				pointB.x = lines[i*8+2];
				pointB.y = lines[i*8+3];
				ab_dir.x = lines[i*8+4];
				ab_dir.y = lines[i*8+5];
				pointC.x = lines[j*8];
				pointC.y = lines[j*8+1];
				pointD.x = lines[j*8+2];
				pointD.y = lines[j*8+3];
				cd_dir.x = lines[j*8+4];
				cd_dir.y = lines[j*8+5];
				polarity = lines[i*8+7];
				if(linearProgram(pointA,pointB,ab_dir,pointC,pointD,cd_dir,polarity,-3*distance_tolerance))//都在彼此的线性区域内
				{
					if(calcCircleParametersAndValidate(lines,line_num,i,j,angles,distance_tolerance,&circle))
					{
						PairedSegmentNode * node = (PairedSegmentNode*)malloc(sizeof(PairedSegmentNode));
						node->circle.x = circle.x;
						node->circle.y = circle.y;
						node->circle.r = circle.r;
						node->pairedSegInd.x = i;//记录是线段对(i,j)
						node->pairedSegInd.y = j;
						//node->next = NULL;//可无
						if(head != NULL)
						{
							tail->next = node;
							tail = node;
						}
						else
						{
							head = tail = node;
						}
						pairlength++; //产生的有效配对数量，也就是有效的候选圆参数数量
					}
				}
				
			}
		}
	if(pairlength > 0)
	{
		PairedSegmentNode *p;
		p = head;
		pairSegmentList = pairedSegmentListInit(pairlength);
		for( i = 0; i<pairSegmentList->length; i++)
		{
			pairSegmentList->pairSeg[i].circle.x = p->circle.x;
			pairSegmentList->pairSeg[i].circle.y = p->circle.y;
			pairSegmentList->pairSeg[i].circle.r = p->circle.r;
			pairSegmentList->pairSeg[i].pairedSegInd.x = p->pairedSegInd.x;//记录线段对(i,j),由lines中的第i条线段和第j条线段构成的匹配对产生该有效圆参数
			pairSegmentList->pairSeg[i].pairedSegInd.y = p->pairedSegInd.y;
			p = p->next;
		}
		tail->next = NULL;
		while (head != NULL)
		{
			p = head;
			head = head->next;
			free(p);
		}
	}
	return pairSegmentList;
}


//===================================================================================================================
//聚类
//求points中一行与initializations中一行里每个元素的平方差总和
double squaredDifference(int & nDims, double *& points, int & i, double *& initializations, int & j)
{
    double result = 0;
    for (int k = 0; k < nDims; ++k)
		result += pow(points[i*nDims+k] - initializations[j*nDims+k], 2);
    return result;
}
/**
 *输入
 *prhs[0]: n x d搜索空间
 *prhs[1]:均值漂移初始化位置，在nxd空间中找均值漂移初始时开始搜索的位置
 *prhs[2]:sigma = 1
 *prhs[3]:window parameter = distance_tolerance或者window parameter = distance_tolerance/2
 *prhs[4]:收敛容忍误差1e-6
 *prhs[5]:迭代次数50
 *输出
 *收敛的位置，位置个数与初始化搜索位置个数一样,我们将结果更新到initPoints,也就是它既是输入参数，也是输出参数，节省内存
 */
void meanShift( double * points, int nPoints, int nDims, double * & initPoints, int initLength, double sigma, double window_size, double accuracy_tolerance, int iter_times )
{
//	for (int i = 0; i<initLength; i++)
//		cout<<initPoints[2*i]<<'\t'<<initPoints[2*i+1]<<endl;
    int nQuerries = initLength;
    double * initializations = (double*)malloc(nQuerries * nDims * sizeof(double));
    memcpy(initializations, initPoints , nQuerries * nDims * sizeof(double));//copy

    double sigma2 = sigma*sigma;//sigma平方
    double radius2 = window_size *window_size;//平方
    double tolerance = accuracy_tolerance;
    int maxiters = iter_times;//最大迭代次数
   //返回与初始搜索点集一样大小的最终定位点集
    double * finals = (double*)malloc(nQuerries * nDims * sizeof(double));;//最终定位点集的指针
    memcpy(finals, initializations, nQuerries * nDims * sizeof(double));
	double * distances = (double*)malloc(nPoints*sizeof(double));
    //printf("meanShift: nPoints:%d \tnDims: %d \tnQuerries:%d \n",nPoints,nDims,nQuerries);//打印
    for (int loop = 0; loop < nQuerries; ++loop)
    {
        int iters = 0;
        while (iters < maxiters)
        {
            bool flag = false;
            double denominator = 0;//分母
            for (int i = 0; i < nPoints; ++i)//对所有的点集进行遍历，找到落在搜索圆域内的点
            {
                distances[i] = squaredDifference(nDims, points, i, initializations, loop);//求距离的平方
                if (distances[i] <= radius2)//在第loop个搜索中心的以sqrt(radius2)为半径的圆域内
                {
                    flag = true;
                    denominator += exp(-distances[i] / sigma2);
                }
            }
            if (!flag)
                break;
            for (int j = 0; j < nDims; ++j)
				finals[loop*nDims+j] = 0;//对最终定位点集中的第loop个点的向量赋值为0
            for (int i = 0; i < nPoints; ++i)
                if (distances[i] <= radius2)
                {
                    for (int j = 0; j < nDims; ++j)//每个内点向量的以一定权值累加
						finals[loop*nDims+j] += exp(-distances[i] / sigma2) * points[i*nDims+j];
                }
            for (int j = 0; j < nDims; ++j)//权值归一化
				finals[loop*nDims+j] /= denominator;
            if (sqrt(squaredDifference(nDims, finals, loop, initializations, loop)) < tolerance)//相继两次的迭代中心在误差内了，则认为已经收敛，没必要再继续迭代
                break;
            iters = iters + 1;
            for (int j = 0; j < nDims; ++j)//更新迭代的搜索中心
				initializations[loop*nDims+j] = finals[loop*nDims+j];
        }
    }
	memcpy(initPoints, finals, nQuerries * nDims * sizeof(double));
    free(distances);
    free(initializations);
	free(finals);
}

/***
 *输入
 *prhs[0] : points,待聚类的点机， n x d
 *prhs[1] ：pointsz中每一个点的靠近的k个点，n x k
 *prhs[2] ：threshold 决定聚类的距离阈值
 *输出 outPoints
 *plhs[0] : 聚类后的店集 m x d 
 */
void clusterByDistance(double * points, int nPoints, int nDims, double distance_threshold,int number_control, double * & outPoints, int * nOutPoints)
{ 
	double threshold2 = distance_threshold*distance_threshold;
    std::vector<double*> centers;
    std::vector<int> counts;
    centers.clear();
    counts.clear();
	bool * labeled = (bool*)malloc(sizeof(bool)*nPoints);
    memset(labeled, 0, nPoints * sizeof(bool));//初始化bool型标签为0
	if(nPoints == 1)
	{
		centers.push_back((double*)malloc(sizeof(double)*nDims));
		for (int k = 0; k < nDims; ++k)
			centers[centers.size() - 1][k] = points[k];
        counts.push_back(1);
	}
	else
	{
		for (int i = 0; i < nPoints-1; ++i)
		{
		    if (!labeled[i])
			{
		        labeled[i] = true;
				centers.push_back((double*)malloc(sizeof(double)*nDims));
		        counts.push_back(1);
		        for (int k = 0; k < nDims; ++k)
				{
				   centers[centers.size() - 1][k] = points[i*nDims+k];  
				}
		        for (int j = i+1; j < nPoints; ++j)
		        {
		            if (!labeled[j])
		            {
		                double d = 0;
		                for (int k = 0; k < nDims; ++k)
				            d += pow(centers[centers.size() - 1][k] / counts[centers.size() - 1] - points[j*nDims+k], 2);
		                if (d <= threshold2)
		                {
		                    ++counts[centers.size() - 1];
		                    for (int k = 0; k < nDims; ++k)
								centers[centers.size() - 1][k] += points[j*nDims+k];
		                    labeled[j] = true;
							if(counts[centers.size() - 1] >= number_control)//聚类数量控制，防止均值中心漂的太远  圆心聚类时20  半径聚类时10
								break;
		                }
		            }
		        }
		    }
		}
	}
    free(labeled);
    centers.shrink_to_fit();
    counts.shrink_to_fit();
    int m = (int) centers.size();
    outPoints = (double*)malloc(sizeof(double)*m*nDims);
	(*nOutPoints) = m;
    for (unsigned int i = 0; i < centers.size(); ++i)
    {
        for (int j = 0; j < nDims; ++j)
		{
			outPoints[i*nDims+j] = centers[i][j] / counts[i];
//			cout<<out[i*nDims+j]<<'\t';
		}
//		cout<<endl;
        free(centers[i]);
    }
    centers.resize(0);
    counts.resize(0);
}
//==============================================================================================================
//获得候选圆心的聚类中心(xi,yi)
//输入：
//pairSegmentList
//输出：
//圆心的聚类中心 centersCandidates是一维double数组， 大小为 centersCandidates_num x 2
void  generateCenterCandidates( PairedSegmentList *pairSegmentList, double distance_tolerance, double *& centerCandidates, int * centerCandidates_num)
{
	double xmax,xmin,ymax,ymin,xdelta,ydelta;
	int nbins_x,nbins_y;
	int x,y;
	int i;
	unsigned int addr;
	xmax = ymax = 0;
	xmin = ymin = DBL_MAX;
	for( i = 0; i< pairSegmentList->length; i++ )
	{
		if( pairSegmentList->pairSeg[i].circle.x > xmax)
			xmax = pairSegmentList->pairSeg[i].circle.x;
		if( pairSegmentList->pairSeg[i].circle.x < xmin)
			xmin = pairSegmentList->pairSeg[i].circle.x;
		if( pairSegmentList->pairSeg[i].circle.y > ymax)
			ymax = pairSegmentList->pairSeg[i].circle.y;
		if( pairSegmentList->pairSeg[i].circle.y < ymin)
			ymin = pairSegmentList->pairSeg[i].circle.y;
	}
	xdelta = (xmax-xmin);
	ydelta = (ymax-ymin);
	nbins_x = (int)ceil(xdelta*2/distance_tolerance);
	nbins_y = (int)ceil(ydelta*2/distance_tolerance);
	if(nbins_x <= 0 || nbins_y <= 0)
	{
		nbins_x = nbins_y = 1;//至少保留1个bin
		//error("generateCircleCandidates,nbins_x,nbins_y error");
	}
	point2d1i * center_bins;
	center_bins = (point2d1i *)malloc(sizeof(point2d1i)*nbins_y*nbins_x);//(x,y,z),x用来记sum(xi),y用来记sum(yi),z用来记落在格子里的数量
	memset(center_bins,0,sizeof(point2d1i)*nbins_y*nbins_x);
	//for( i = 0; i<nbins_x*nbins_y; i++)
	//{
	//	center_bins[i].x = center_bins[i].y = 0;
	//	center_bins[i].z = 0;
	//}
	if(center_bins == NULL)
		error("generateCircleCandidates,not enough memory");
	for ( i = 0; i< pairSegmentList->length; i++ )//将圆心记录到格子里面，同时落在相应格子里面的数量++
	{
		x = (int)((pairSegmentList->pairSeg[i].circle.x - xmin)/xdelta*nbins_x+0.5);//四舍五入
		y = (int)((pairSegmentList->pairSeg[i].circle.y - ymin)/ydelta*nbins_y+0.5);
		x = x<0 ? 0 :x;
		x = x>= nbins_x ? (nbins_x-1):x;
		y = y<0 ? 0 :y;
		y = y>= nbins_y ? (nbins_y-1):y;
		addr = y*nbins_x+x;
		center_bins[addr].x += pairSegmentList->pairSeg[i].circle.x;
		center_bins[addr].y += pairSegmentList->pairSeg[i].circle.y;
		center_bins[addr].z ++;
	}
	Point2dNode * head, * tail ;
	int initCentersLength = 0;
	head = tail = NULL;
	for ( y = 0; y<nbins_y; y++)//将vote后非0的格子里面的圆心取均值，并记录到链表里面
		for ( x = 0; x<nbins_x; x++)
		{
			addr = y*nbins_x+x;
			if(center_bins[addr].z > 0)
			{
				Point2dNode * node = (Point2dNode*)malloc(sizeof(Point2dNode));
				node->point.x = center_bins[addr].x/center_bins[addr].z;
				node->point.y = center_bins[addr].y/center_bins[addr].z;
				initCentersLength++;
				if(head != NULL)
				{
					tail->next = node;
					tail = node;
					tail->next = NULL;//可无
				}
				else
				{
					head = tail = node;
				}
			}
		}
	if(initCentersLength == 0)
	{
		(*centerCandidates_num) = 0;
		centerCandidates = NULL;
		//error("generateCircleCandidates,initCentersLength equals 0");
	}
	free(center_bins);//赶紧释放该内存
	double * initCenters; //initCentersLength x 2
	initCenters = (double*)malloc(sizeof(double)*initCentersLength*2); 
	tail = head;//利用tail
	//将记录在链表里面的分区后的圆心均值记录到数组里，便于作为初始点进行均值漂移
	for ( i = 0; i<initCentersLength; i++ )// initCenters 大小是 initCentersLength*2
	{
		//一边记录到double数组，一边释放内存
		int addr = 2*i;
		initCenters[addr] = tail->point.x;
		initCenters[addr+1] = tail->point.y;
		head = tail;       
		tail = tail->next;
		free(head);
	}
//	for (int  i = 0; i<initCentersLength; i++)
//		cout<<initCenters[2*i]<<'\t'<<initCenters[2*i+1]<<endl;

	double * originCenters;
	originCenters = (double*)malloc(sizeof(double)*pairSegmentList->length*2); //原始的圆心集合数量 x 2
	for ( i = 0; i<pairSegmentList->length; i++)//原始圆心集合复制到double数组中
	{
		originCenters[i]   = pairSegmentList->pairSeg[i].circle.x;
		originCenters[i+1] = pairSegmentList->pairSeg[i].circle.y;
	}
	//均值漂移的结果会更新到initCenters里面
	meanShift(originCenters,pairSegmentList->length,2,initCenters,initCentersLength,1,distance_tolerance/2,1e-6,50);
//	for (int  i = 0; i<initCentersLength; i++)
//		cout<<initCenters[2*i]<<'\t'<<initCenters[2*i+1]<<endl;
	//聚类
//	double * centerCandidates;//== candidateCenters_num x 2    (xi,yi)
//	int centerCandidates_num;
	free(originCenters);//释放内存
	//千万要注意centerCandidates_num是int型指针，++--时要(*centerCandidates_num).
	clusterByDistance(initCenters,initCentersLength,2,distance_tolerance,20,centerCandidates, centerCandidates_num);
//	cout<<"candidateCenters_num(圆心均值漂移和聚类后数量):"<<(*centerCandidates_num)<<endl;
//	return centerCandidates;
}

void  generateCircleCandidates( PairedSegmentList *pairSegmentList, double * centerCandidates, int centerCandidates_num, double distance_tolerance, double *& circleCandidates, int * circleCandidates_num)
{
	Point3dNode * circleCandidates_head,*circleCandidates_tail;//将产生的候选圆参数(x,y,r)先存储在链表里，然后再放到一维数组circleCandidates中，circleCandidates_num x 3
	circleCandidates_head = circleCandidates_tail = NULL; //初始化为NULL
	(*circleCandidates_num) = 0;//产生的候选圆数量初始化为0

	double dismin,temp;
	int ind;
	int i,j;
	Point1dNode ** head, **tail;
	int * count;
	head = (Point1dNode**)malloc(sizeof(Point1dNode*)*centerCandidates_num);
	tail = (Point1dNode**)malloc(sizeof(Point1dNode*)*centerCandidates_num);
	count = (int *)malloc(sizeof(int)*centerCandidates_num);
	memset(head,NULL,sizeof(Point1dNode*)*centerCandidates_num);//初始化为NULL
	memset(tail,NULL,sizeof(Point1dNode*)*centerCandidates_num);
	memset(count,0,sizeof(int)*centerCandidates_num);//初始化为0
	//首先要将初始的圆心集合按照聚类后的中心分类，得到每个聚类中心可能的半径
	for ( i = 0; i<pairSegmentList->length; i++)
	{
	    dismin = DBL_MAX;
		ind    = -1;
		for ( j = 0; j<centerCandidates_num; j++)
		{
			temp = dist(pairSegmentList->pairSeg[i].circle.x , pairSegmentList->pairSeg[i].circle.y, centerCandidates[2*j],centerCandidates[2*j+1]);
			if(temp < dismin)
			{
				ind = j;
				dismin = temp;
			}
		}
		Point1dNode * node = (Point1dNode*)malloc(sizeof(Point1dNode));
		node->data =pairSegmentList->pairSeg[i].circle.r;
		count[ind]++;  //计数
		if(head[ind] != NULL)
		{
			tail[ind]->next = node;
			tail[ind] = node;
			//tail[ind]->next = NULL;
		}
		else
			head[ind] = tail[ind] = node;
	}
	//对每一个聚类圆心的可能的半径进行聚类
	for ( i = 0; i<centerCandidates_num; i++ )
	{
		int      origin_r_length = count[i];
		double * origin_r = (double*)malloc(sizeof(double)*origin_r_length);
		Point1dNode * p,*releasep;
		double rmin,rmax,rdelta;
		rmin = DBL_MAX;
		rmax = 0;
		p = head[i];
		for( j  = 0; j < origin_r_length; j++)//将链表里的r集合复制到数组
		{
			if(p->data < rmin)//在这一次遍历中，记录最大最小值
				rmin = p->data;
			if(p->data > rmax)
				rmax = p->data;
			origin_r[j] = p->data;
			releasep = p;
			p = p->next;
			free(releasep);//一边分区，一边释放一个聚类圆心对应的r集合时申请的内存
		}
		//这样我们便得到在圆心聚类中心(xi,yi) = (centerCandidates[2*i],centerCandidates[2*i+1])处可能的r集合 r[r_length].
		int nbins_r = 0;
		point1d1i * r_bins;
		rmax += rmin*0.05;//避免rmax-rmin = 0
		rmin -= rmin*0.05;
		rdelta = rmax - rmin;
		nbins_r = (int)ceil(2*(rdelta)/distance_tolerance); //此处与原论文的有不同
	    if(nbins_r <= 0)//至少有一个bin
			nbins_r = 1;
		r_bins = (point1d1i *)malloc(sizeof(point1d1i)*nbins_r);
		memset(r_bins,0,sizeof(point1d1i)*nbins_r);//初始化为0
		for( j = 0; j<origin_r_length; j++)//对分区vote
		{
			ind = int((origin_r[j]-rmin)/rdelta*nbins_r+0.5);
			ind = ind < 0 ? 0 : ind;
			ind = ind >= nbins_r ? (nbins_r-1) : ind;
			r_bins[ind].data += origin_r[j];
			r_bins[ind].cnt  ++;			
		}
		int init_r_length = 0;
		double * init_r ;//= (double*)malloc(sizeof(double)*r_length);
		for( j = 0; j<nbins_r; j++)
		{
			if(r_bins[j].cnt > 0)//统计非0分区
			{
				init_r_length++;
			}
		}
		init_r = (double*)malloc(sizeof(double)*init_r_length);
		ind = 0;
		for( j = 0; j<nbins_r; j++)
		{
			if(r_bins[j].cnt > 0)//将均值移动到数组里，此处的比较重复了2遍,如何优化？
			{
				init_r[ind++] = r_bins[j].data/r_bins[j].cnt;  //取均值
			}
		}
		free(r_bins);//释放分区时申请的内存
		//至此，我们的到了均值漂移初始的ri，为一维double数组init_r，长度是init_r_len
		meanShift(origin_r,origin_r_length,1,init_r,init_r_length,1,distance_tolerance/2,1e-6,50);
		free(origin_r);//释放第i个聚类圆心对应的原始r集合
		double * r_candidates;
		int r_candidates_num;
		clusterByDistance(init_r,init_r_length,1,distance_tolerance,10,r_candidates,&r_candidates_num);
		if(r_candidates_num <= 0)//可无
		{
			continue;  //不懂为什么，聚类中心的周围确没有最靠近它的点
			//error("generateCircleCandidates,r_candidates_num<=0");
		}
		for( j = 0; j<r_candidates_num; j++)
		{
			Point3dNode * node = (Point3dNode *)malloc(sizeof(Point3dNode));
			node->point.x = centerCandidates[2*i];
			node->point.y = centerCandidates[2*i+1];
			node->point.r = r_candidates[j];
			if(circleCandidates_head != NULL)
			{
				circleCandidates_tail->next = node;
				circleCandidates_tail = node;
				(*circleCandidates_num)++;  //候选圆组合数量++
			}
			else
				circleCandidates_head = circleCandidates_tail = node;
		}

		free(r_candidates);//释放候选r
	}
	free(head);
	free(tail);
	free(count);
	
//	cout<<"候选圆组合数量为："<<*circleCandidates_num<<endl;
	//千万要注意circleCandidates_num是int型指针，++--时要(*circleCandidates_num).
	circleCandidates = (double*)malloc(sizeof(double)*(*circleCandidates_num)*3);
    //将候选圆从链表中复制到double数组circleCandidates中
	for( i = 0; i < (*circleCandidates_num); i++)
	{
		int addr = 3*i;
		circleCandidates[addr]   = circleCandidates_head->point.x;
		circleCandidates[addr+1] = circleCandidates_head->point.y;
		circleCandidates[addr+2] = circleCandidates_head->point.r;
		circleCandidates_tail = circleCandidates_head;
		circleCandidates_head = circleCandidates_head->next;
		free(circleCandidates_tail);//一边转移，一边释放候选圆链表
	}
/*
	fstream file1;
	file1.open("D:\\Graduate Design\\picture\\sp\\circlecandidates.txt",ios::out | ios::trunc);
	file1.close();
	file1.open("D:\\Graduate Design\\picture\\sp\\circlecandidates.txt",ios::out | ios::app);
	file1<<"the  combination center and radii(circle candidates)num is :"<<*circleCandidates_num<<endl;
	for( i = 0; i < (*circleCandidates_num); i++)
	{
		int addr = 3*i;
		file1<<circleCandidates[addr]<<'\t'<<circleCandidates[addr+1]<<'\t'<<circleCandidates[addr+2]<<endl;
	}
	file1.close();
*/
}
//==========================================END=======================================================================
/**
输入：
prhs[0]: 输入的灰度图像，单通道，大小是imgy x imgx
输出：
plhs[0]: 候选圆组合(xi,yi,ri)', 3 x m
plhs[1]: 边缘图，大小是imgy x imgx，设边缘点总数为 edgepix_n
plhs[2]: 边缘点的梯度向量矩阵，大小是 2 x edgepix_n
plhs[3]: 线段图，大小是imgy x imgx 
*/
//mex generateCircleCandidates.cpp -IF:\OpenCV\opencv2.4.9\build\include -IF:\OpenCV\opencv2.4.9\build\include\opencv -IF:\OpenCV\opencv2.4.9\build\include\opencv2 -LF:\OpenCV\opencv2.4.9\build\x64\vc11\lib -lopencv_core249 -lopencv_highgui249 -lopencv_imgproc249
//======================================MEX function==================================================================

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if(nrhs!=1) 
      mexErrMsgIdAndTxt( "MATLAB:revord:invalidNumInputs","One input required.");
    else if(nlhs > 4) 
      mexErrMsgIdAndTxt( "MATLAB:revord:maxlhs","Too many output arguments.");
	unsigned char * inputimg = (unsigned char*)mxGetData(prhs[0]);
	int imgy,imgx;
	imgy = (int)mxGetM(prhs[0]);
	imgx = (int)mxGetN(prhs[0]);
	double *data=(double*)malloc(imgy*imgx*sizeof(double));//将输入矩阵中的图像数据转存到一维数组中
    for(int c=0;c<imgx;c++)
    {
        for(int r=0;r<imgy;r++)
        {
           data[c+r*imgx]=inputimg[r+c*imgy];              
        }    
    }
	int n;//线段数量
    double* out=mylsd(&n, data,imgx,imgy,NULL,NULL,NULL);
    printf("LSD,output ls num: %i\n",n);

	 image_double angles;
	 calculateGradient2(data,imgx,imgy,&angles);
	 PairedSegmentList * pairseglist;
	 double distance_tolerance = max( 2.0, 0.005*min(angles->xsize,angles->ysize) ); // 0.005%*min(xsize,ysize)
	 pairseglist = getValidatePairedSegment(out,n,angles,distance_tolerance);
	 if(pairseglist != NULL)
	 {
		printf("有效线段对数量：%i \n",pairseglist->length);
		double * centers;
		int      centers_num;
		generateCenterCandidates(pairseglist,distance_tolerance,centers,&centers_num);
		printf("圆心均值漂移并聚类后的圆心数量：%i \n",centers_num);
		double * candidates;
		int  candidates_num;
		generateCircleCandidates(pairseglist,centers,centers_num,distance_tolerance,candidates,&candidates_num);
		printf("半径均值漂移并聚类后，所有的候选圆组合数量：%i \n",candidates_num);
		
		double *candidates_out;
		plhs[0] = mxCreateDoubleMatrix(3,candidates_num,mxREAL);
		candidates_out = (double*)mxGetPr(plhs[0]);
		//候选圆组合(xi,yi,ri)',3 x candidates_num, 复制到矩阵candidates_out中
		memcpy(candidates_out,candidates,sizeof(double)*3*candidates_num);

		free(pairseglist->pairSeg);
		free(centers);
		free(candidates);
	 }
	 else
	 {
		 printf("有效线段对数量：%i \n",0);
		 double *candidates_out;
		 plhs[0] = mxCreateDoubleMatrix(3,1,mxREAL);
		 candidates_out = (double*)mxGetPr(plhs[0]);
		 candidates_out[0] = candidates_out[1] = candidates_out[2] = 0;
	 }
	 unsigned char *edgeimg_out;
	 unsigned int edge_pixels_total_num = 0;//边缘总像素
	 double *gradient_vec_out;
	 plhs[1] = mxCreateNumericMatrix(imgy,imgx,mxUINT8_CLASS,mxREAL);
	 edgeimg_out = (unsigned char*)mxGetData(plhs[1]);
	 //将边缘图复制到矩阵edgeimg_out中
	 //将梯度向量存到矩阵gradient_vec_out中
	 unsigned int addr,g_cnt = 0;
	 for ( int c = 0; c < imgx; c++ )
		 for ( int r = 0; r < imgy; r++)
		 {
			 addr = r*imgx+c;
			 if(angles->data[addr] == NOTDEF)
				 edgeimg_out[c*imgy+r] = 0;
			 else
			 {
				 edgeimg_out[c*imgy+r] = 255;//为边缘点，赋值为白色
				 //------------------------------------------------
				 edge_pixels_total_num++;
			 }
		 }
	//申请edge_pixels_total_num x 2 来保存每一个边缘点的梯度向量，以列为优先，符合matlab的习惯
	 plhs[2] = mxCreateDoubleMatrix(2,edge_pixels_total_num,mxREAL);
	 gradient_vec_out = (double*)mxGetPr(plhs[2]);
	  for ( int c = 0; c < imgx; c++ )
		 for ( int r = 0; r < imgy; r++)
		 {
			 addr = r*imgx+c;
			 if(angles->data[addr] != NOTDEF)
			 {
				 gradient_vec_out[g_cnt++] = cos(angles->data[addr]);
				 gradient_vec_out[g_cnt++] = sin(angles->data[addr]);
			 }
		 }
	 //---------------------------------------------------------------------
	//输出线段检测的图像
	//if(nlhs == 4)
	//{
	//	Mat ls_mat = Mat::zeros(imgy,imgx,CV_8UC1);
	//	for ( int i = 0; i<n ; i++)
	//	{
	//	  Point2d p1(out[8*i],out[8*i+1]),p2(out[8*i+2],out[8*i+3]);
	//	  line(ls_mat,p1,p2,Scalar(255,0,0));
	//	}
	//	plhs[3] = mxCreateDoubleMatrix(imgy,imgx,mxREAL);
	//	double * ls_img_out = (double*)mxGetPr(plhs[3]);
	//	//memcpy(ls_out_mat,ls_mat.data ,sizeof(unsigned char)*M*N);
	//	for (int i = 0; i<imgx; i++)
	//		for (int j = 0; j<imgy;j++)
	//			ls_img_out[i*imgy+j]=ls_mat.data[j*imgx+i];
	//}
	//---------------------------------------------------------------------
	//这里的free是释放程序中用于产生候选圆所用到的一系列内存
	free(data);
	free(out);
	free_image_double(angles);
}





