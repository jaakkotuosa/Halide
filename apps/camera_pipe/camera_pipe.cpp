#include <FImage.h>
#include <stdint.h>

using namespace FImage;

Var x, y, tx, ty, c;

Func hot_pixel_suppression(Func input) {
    Expr max = Max(Max(input(x-2, y), input(x+2, y)),
                   Max(input(x, y-2), input(x, y+2)));
    Expr min = Min(Min(input(x-2, y), input(x+2, y)),
                   Min(input(x, y-2), input(x, y+2)));
    
    Func denoised("denoised");
    denoised(x, y) = Clamp(input(x, y), min, max);
    
    // Compute this in chunks over tiles, vectorized by 8
    denoised.chunk(tx).vectorize(x, 8);

    return denoised;
}

Expr abs(Expr e) {
    return Select(e < Cast(e.type(), 0), -e, e);
}

Func interleave_x(Func a, Func b) {
    Func out;
    out(x, y) = Select((x%2)==0, a(x/2, y), b(x/2, y));
    return out;
}

Func interleave_y(Func a, Func b) {
    Func out;
    out(x, y) = Select((y%2)==0, a(x, y/2), b(x, y/2));
    return out;
}

Func demosaic(Func raw) {
    // These are the values we already know from the input
    // x_y = the value of channel x at a site in the input of channel y
    // gb refers to green sites in the blue rows
    // gr refers to green sites in the red rows

    // Deinterleave the color channels
    Func deinterleaved("deinterleaved");
    Var t, ti;
    deinterleaved(x, y, t) = Select(t == 0, raw(2*x, 2*y), 
                                    Select(t == 1, raw(2*x+1, 2*y), 
                                           Select(t == 2, raw(2*x, 2*y+1), raw(2*x+1, 2*y+1))));

    // Unroll the tuple part and compute it in the inside loop
    deinterleaved.split(t, t, ti, 4).unroll(ti).transpose(y, ti).transpose(x, ti);
    deinterleaved.chunk(tx).vectorize(x, 8);

    // Give more convenient names to the four channels we know
    Func r_r("r_r"), g_gr("g_gr"), g_gb("g_gb"), b_b("b_b");    
    g_gr(x, y) = deinterleaved(x, y, 0);
    r_r(x, y)  = deinterleaved(x, y, 1);
    b_b(x, y)  = deinterleaved(x, y, 2);
    g_gb(x, y) = deinterleaved(x, y, 3);

    // These are the ones we need to interpolate
    Func b_r("b_r"), g_r("g_r");
    Func b_gr("b_gr"), r_gr("r_gr");
    Func b_gb("b_gb"), r_gb("r_gb");
    Func r_b("r_b"), g_b("g_b");

    // First calculate green at the red and blue sites

    // Try interpolating vertically and horizontally. Also compute
    // differences vertically and horizontally. Use interpolation in
    // whichever direction had the smallest difference.
    Expr gv_r  =    (g_gb(x, y-1) + g_gb(x, y))/2;
    Expr gvd_r = abs(g_gb(x, y-1) - g_gb(x, y));
    Expr gh_r  =    (g_gr(x+1, y) + g_gr(x, y))/2;
    Expr ghd_r = abs(g_gr(x+1, y) - g_gr(x, y));

    g_r(x, y)  = Select(ghd_r < gvd_r, gh_r, gv_r);

    Expr gv_b  =    (g_gr(x, y+1) + g_gr(x, y))/2;
    Expr gvd_b = abs(g_gr(x, y+1) - g_gr(x, y));
    Expr gh_b  =    (g_gb(x-1, y) + g_gb(x, y))/2;
    Expr ghd_b = abs(g_gb(x-1, y) - g_gb(x, y));

    g_b(x, y)  = Select(ghd_b < gvd_b, gh_b, gv_b);

    // Next interpolate red at gr by first interpolating, then
    // correcting using the error green would have had if we had
    // interpolated it in the same way (i.e. add the second derivative
    // of the green channel at the same place).
    Expr correction;
    correction = g_gr(x, y) - (g_r(x, y) + g_r(x-1, y))/2;
    r_gr(x, y) = correction + (r_r(x-1, y) + r_r(x, y))/2;

    // Do the same for other reds and blues at green sites
    correction = g_gr(x, y) - (g_b(x, y) + g_b(x, y-1))/2;
    b_gr(x, y) = correction + (b_b(x, y) + b_b(x, y-1))/2;

    correction = g_gb(x, y) - (g_r(x, y) + g_r(x, y+1))/2;
    r_gb(x, y) = correction + (r_r(x, y) + r_r(x, y+1))/2;

    correction = g_gb(x, y) - (g_b(x, y) + g_b(x+1, y))/2;
    b_gb(x, y) = correction + (b_b(x, y) + b_b(x+1, y))/2;

    // Now interpolate diagonally to get red at blue and blue at
    // red. Hold onto your hats; this gets really fancy. We do the
    // same thing as for interpolating green where we try both
    // directions (in this case the positive and negative diagonals),
    // and use the one with the lowest absolute difference. But we
    // also use the same trick as interpolating red and blue at green
    // sites - we correct our interpolations using the second
    // derivative of green at the same sites.

    correction = g_b(x, y)  - (g_r(x, y) + g_r(x-1, y+1))/2;
    Expr rp_b  = correction + (r_r(x, y) + r_r(x-1, y+1))/2;
    Expr rpd_b = abs(r_r(x, y) - r_r(x-1, y+1));

    correction = g_b(x, y)  - (g_r(x-1, y) + g_r(x, y+1))/2;
    Expr rn_b  = correction + (r_r(x-1, y) + r_r(x, y+1))/2;
    Expr rnd_b = abs(r_r(x-1, y) - r_r(x, y+1));

    r_b(x, y)  = Select(rpd_b < rnd_b, rp_b, rn_b);


    // Same thing for blue at red
    correction = g_r(x, y)  - (g_b(x, y) + g_b(x+1, y-1))/2;
    Expr bp_r  = correction + (b_b(x, y) + b_b(x+1, y-1))/2;
    Expr bpd_r = abs(b_b(x, y) - b_b(x+1, y-1));

    correction = g_r(x, y)  - (g_b(x+1, y) + g_b(x, y-1))/2;
    Expr bn_r  = correction + (b_b(x+1, y) + b_b(x, y-1))/2;
    Expr bnd_r = abs(b_b(x+1, y) - b_b(x, y-1));

    b_r(x, y)  =  Select(bpd_r < bnd_r, bp_r, bn_r);    

    // Interleave the resulting channels
    Func r = interleave_y(interleave_x(r_gr, r_r),
                          interleave_x(r_b, r_gb));
    Func g = interleave_y(interleave_x(g_gr, g_r),
                          interleave_x(g_b, g_gb));
    Func b = interleave_y(interleave_x(b_gr, b_r),                          
                          interleave_x(b_b, b_gb));

    // Compute these in chunks over tiles, vectorized by 8
    g_r.chunk(tx).vectorize(x, 8);
    g_b.chunk(tx).vectorize(x, 8);
    r_gr.chunk(tx).vectorize(x, 8);
    b_gr.chunk(tx).vectorize(x, 8);
    r_gb.chunk(tx).vectorize(x, 8);
    b_gb.chunk(tx).vectorize(x, 8);
    r_b.chunk(tx).vectorize(x, 8);
    b_r.chunk(tx).vectorize(x, 8);
    // These interleave in y, so unrolling them in y might help
    r.chunk(tx).vectorize(x, 8).unroll(y, 2);
    g.chunk(tx).vectorize(x, 8).unroll(y, 2);
    b.chunk(tx).vectorize(x, 8).unroll(y, 2);

    Func output("dem");
    output(x, y) = (r(x, y), g(x, y), b(x, y));

    return output;
}


Func color_correct(Func input, UniformImage matrix_3200, UniformImage matrix_7000, Uniform<float> kelvin) {
    // Get a color matrix by linearly interpolating between two
    // calibrated matrices using inverse kelvin.

    Func matrix("matrix");
    Expr alpha = (1.0f/kelvin - 1.0f/3200) / (1.0f/7000 - 1.0f/3200);
    Expr val =  (matrix_3200(x, y) * alpha + matrix_7000(x, y) * (1 - alpha));
    matrix(x, y) = Cast<int32_t>(val * 256.0f); // Q8.8 fixed point
    matrix.root();

    Func corrected("corrected");
    Expr ir = Cast<int32_t>(input(x, y, 0));
    Expr ig = Cast<int32_t>(input(x, y, 1));
    Expr ib = Cast<int32_t>(input(x, y, 2));
   
    Expr r, g, b;
    r = matrix(3, 0) + matrix(0, 0) * ir + matrix(1, 0) * ig + matrix(2, 0) * ib;
    g = matrix(3, 1) + matrix(0, 1) * ir + matrix(1, 1) * ig + matrix(2, 1) * ib;
    b = matrix(3, 2) + matrix(0, 2) * ir + matrix(1, 2) * ig + matrix(2, 2) * ib;

    corrected(x, y) = Cast<int16_t>((r, g, b)/256);

    Var c = corrected.arg(2);
    corrected.chunk(tx).transpose(y, c).transpose(x, c).vectorize(x, 4).unroll(c, 3);
    
    return corrected;
}


Func apply_curve(Func input, Uniform<float> gamma, Uniform<float> contrast) {
    // copied from FCam
    Func curve("curve");

    Expr xf = Clamp(Cast<float>(x)/1024.0f, 0.0f, 1.0f);
    Expr g = pow(xf, 1.0f/gamma);
    Expr b = 2.0f - pow(2.0f, contrast/100.0f);
    Expr a = 2.0f - 2.0f*b; 
    Expr z = Select(g > 0.5f,
                    1.0f - (a*(1.0f-g)*(1.0f-g) + b*(1.0f-g)),
                    a*g*g + b*g);
    Expr val = Cast<uint8_t>(Clamp(z*256.0f, 0.0f, 255.0f));
    curve(x) = val;
    curve.root(); // It's a LUT, compute it once ahead of time.

    Func curved("curved");
    curved(x, y, c) = curve(input(x, y, c));

    return curved;
}

Func process(Func raw, 
             UniformImage matrix_3200, UniformImage matrix_7000, Uniform<float> color_temp, 
             Uniform<float> gamma, Uniform<float> contrast) {

    Func processed("p");
    Var xi, yi;

    Func im = raw;
    im = hot_pixel_suppression(im);
    im = demosaic(im);
    im = color_correct(im, matrix_3200, matrix_7000, color_temp);
    im = apply_curve(im, gamma, contrast);

    Var co, ci;    
    processed(tx, ty, c) = im(tx, ty, ci);

    processed.split(c, co, ci, 3);
    processed.tile(tx, ty, xi, yi, 32, 32).transpose(ty, ci).transpose(tx, ci);//.vectorize(xi, 8).parallel(ty);

    return processed;
}

int main(int argc, char **argv) {
    // The camera pipe is specialized on the 2592x1968 images that
    // come in, so we'll just use an image instead of a uniform image.
    Image<int16_t> input(2592, 1968);
    UniformImage matrix_3200(Float(32), 2, "m3200"), matrix_7000(Float(32), 2, "m7000");
    Uniform<float> color_temp("color_temp", 3200.0f);
    Uniform<float> gamma("gamma", 1.8f);
    Uniform<float> contrast("contrast", 10.0f);

    // shift things inwards to give us enough padding on the
    // boundaries so that we don't need to check bounds. We're going
    // to make a 2560x1920 output image, so shift by 16, 12
    Func clamped("in");
    clamped(x, y) = input(x+16, y+12);

    // Run the pipeline
    Func processed = process(clamped, matrix_3200, matrix_7000, color_temp, gamma, contrast);

    processed.compileToFile("curved");
    
    return 0;
}

