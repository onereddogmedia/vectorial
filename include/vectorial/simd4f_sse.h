/*
  Vectorial
  Copyright (c) 2010 Mikko Lehtonen
  Licensed under the terms of the two-clause BSD License (see LICENSE)
*/
#ifndef VECTORIAL_SIMD4F_SSE_H
#define VECTORIAL_SIMD4F_SSE_H

#include <xmmintrin.h>
#include <string.h>  // memcpy

#ifdef __cplusplus
extern "C" {
#endif


typedef __m128 simd4f; 
typedef __m128i simd4i; 

typedef union {
    simd4f s ;
    float f[4];
    unsigned int ui[4];
} _simd4f_union;

// creating

vectorial_inline simd4f simd4f_create(const float& x, const float& y, const float& z, const float& w) {
    simd4f s = { x, y, z, w };
    return s;
}

vectorial_inline simd4f simd4i_create(const int& x, const int& y, const int& z, const int& w) {
    simd4i s = { (long long)x << 32 | y, (long long)z << 32 | w };
    return s;
}

vectorial_inline simd4f simd4f_zero() { return _mm_setzero_ps(); }

vectorial_inline simd4i simd4i_zero() { return _mm_setzero_si128(); }

vectorial_inline simd4f simd4f_uload4(const float *ary) {
    simd4f s = _mm_loadu_ps(ary);
    return s;
}

vectorial_inline simd4f simd4f_uload3(const float *ary) {
    simd4f s = simd4f_create(ary[0], ary[1], ary[2], 0);
    return s;
}

vectorial_inline simd4f simd4f_uload2(const float *ary) {
    simd4f s = simd4f_create(ary[0], ary[1], 0, 0);
    return s;
}


vectorial_inline void simd4f_ustore4(const simd4f& val, float *ary) {
    _mm_storeu_ps(ary, val);
}

vectorial_inline void simd4f_ustore3(const simd4f& val, float *ary) {
    memcpy(ary, &val, sizeof(float) * 3);
}

vectorial_inline void simd4f_ustore2(const simd4f& val, float *ary) {
    memcpy(ary, &val, sizeof(float) * 2);
}


// utilites

vectorial_inline simd4f simd4f_splat(const float& v) { 
    simd4f s = _mm_set1_ps(v); 
    return s;
}

vectorial_inline simd4f simd4f_splat_x(const simd4f& v) { 
    simd4f s = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0,0,0,0)); 
    return s;
}

vectorial_inline simd4f simd4f_splat_y(const simd4f& v) { 
    simd4f s = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1,1,1,1)); 
    return s;
}

vectorial_inline simd4f simd4f_splat_z(const simd4f& v) { 
    simd4f s = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2,2,2,2)); 
    return s;
}

vectorial_inline simd4f simd4f_splat_w(const simd4f& v) { 
    simd4f s = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3,3,3,3)); 
    return s;
}

vectorial_inline simd4i simd4i_splat(const int& v) { 
    simd4i s = _mm_set1_epi32(v);
    return s;
}

vectorial_inline simd4i simd4f_convert(const simd4f& v) { 
    simd4i s = _mm_cvttps_epi32(v);
    return s;
}

vectorial_inline simd4f simd4i_convert(const simd4i& v) { 
    simd4i s = _mm_cvtepi32_ps(v);
    return s;
}



// arithmetic

vectorial_inline simd4f simd4f_add(const simd4f& lhs, const simd4f& rhs) {
    simd4f ret = _mm_add_ps(lhs, rhs);
    return ret;
}

vectorial_inline simd4f simd4f_sub(const simd4f& lhs, const simd4f& rhs) {
    simd4f ret = _mm_sub_ps(lhs, rhs);
    return ret;
}

vectorial_inline simd4f simd4f_mul(const simd4f& lhs, const simd4f& rhs) {
    simd4f ret = _mm_mul_ps(lhs, rhs);
    return ret;
}

vectorial_inline simd4f simd4f_div(const simd4f& lhs, const simd4f& rhs) {
    simd4f ret = _mm_div_ps(lhs, rhs);
    return ret;
}

vectorial_inline simd4f simd4f_madd(const simd4f& m1, const simd4f& m2, const simd4f& a) {
    return simd4f_add( simd4f_mul(m1, m2), a );
}

vectorial_inline simd4i simd4i_add(const simd4i& lhs, const simd4i& rhs) {
    simd4i ret = _mm_add_epi32(lhs, rhs);
    return ret;
}

static const __m128 SIGNMASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
               
vectorial_inline simd4f simd4f_neg(const simd4f& v) { 
    simd4i ret = _mm_xor_ps(v, SIGNMASK);
    return ret;
}


vectorial_inline simd4f simd4f_reciprocal(const simd4f& v) { 
    simd4f s = _mm_rcp_ps(v); 
    const simd4f two = simd4f_create(2.0f, 2.0f, 2.0f, 2.0f);
    s = simd4f_mul(s, simd4f_sub(two, simd4f_mul(v, s)));
    return s;
}

vectorial_inline simd4f simd4f_sqrt(const simd4f& v) { 
    simd4f s = _mm_sqrt_ps(v); 
    return s;
}

vectorial_inline simd4f simd4f_rsqrt(const simd4f& v) { 
    simd4f s = _mm_rsqrt_ps(v); 
    const simd4f half = simd4f_create(0.5f, 0.5f, 0.5f, 0.5f);
    const simd4f three = simd4f_create(3.0f, 3.0f, 3.0f, 3.0f);
    s = simd4f_mul(simd4f_mul(s, half), simd4f_sub(three, simd4f_mul(s, simd4f_mul(v,s))));
    return s;
}




vectorial_inline simd4f simd4f_cross3(const simd4f& lhs, const simd4f& rhs) {
    
    const simd4f lyzx = _mm_shuffle_ps(lhs, lhs, _MM_SHUFFLE(3,0,2,1));
    const simd4f lzxy = _mm_shuffle_ps(lhs, lhs, _MM_SHUFFLE(3,1,0,2));

    const simd4f ryzx = _mm_shuffle_ps(rhs, rhs, _MM_SHUFFLE(3,0,2,1));
    const simd4f rzxy = _mm_shuffle_ps(rhs, rhs, _MM_SHUFFLE(3,1,0,2));

    return _mm_sub_ps(_mm_mul_ps(lyzx, rzxy), _mm_mul_ps(lzxy, ryzx));

}



vectorial_inline float simd4f_get_x(const simd4f& s) { _simd4f_union u={s}; return u.f[0]; }
vectorial_inline float simd4f_get_y(const simd4f& s) { _simd4f_union u={s}; return u.f[1]; }
vectorial_inline float simd4f_get_z(const simd4f& s) { _simd4f_union u={s}; return u.f[2]; }
vectorial_inline float simd4f_get_w(const simd4f& s) { _simd4f_union u={s}; return u.f[3]; }

vectorial_inline int simd4i_get_x(const simd4i& s) { _simd4f_union u={s}; return u.ui[0]; }
vectorial_inline int simd4i_get_y(const simd4i& s) { _simd4f_union u={s}; return u.ui[1]; }
vectorial_inline int simd4i_get_z(const simd4i& s) { _simd4f_union u={s}; return u.ui[2]; }
vectorial_inline int simd4i_get_w(const simd4i& s) { _simd4f_union u={s}; return u.ui[3]; }


vectorial_inline simd4f simd4f_shuffle_wxyz(const simd4f& s) { return _mm_shuffle_ps(s,s, _MM_SHUFFLE(2,1,0,3) ); }
vectorial_inline simd4f simd4f_shuffle_zwxy(const simd4f& s) { return _mm_shuffle_ps(s,s, _MM_SHUFFLE(1,0,3,2) ); }
vectorial_inline simd4f simd4f_shuffle_yzwx(const simd4f& s) { return _mm_shuffle_ps(s,s, _MM_SHUFFLE(0,3,2,1) ); }

vectorial_inline simd4f simd4f_zero_w(simd4f s) {
    simd4f r = _mm_unpackhi_ps(s, _mm_setzero_ps());
    return _mm_movelh_ps(s, r);
}

vectorial_inline simd4f simd4f_zero_zw(simd4f s) {
    return _mm_movelh_ps(s, _mm_setzero_ps());
}

vectorial_inline simd4f simd4f_merge_high(simd4f xyzw, simd4f abcd) { 
    return _mm_movehl_ps(abcd, xyzw);
}


typedef simd4f_aligned16 union {
    unsigned int ui[4];
    float f[4];
} _simd4f_uif;

vectorial_inline simd4f simd4f_flip_sign_0101(const simd4f& s) {
    const _simd4f_uif upnpn = { { 0x00000000, 0x80000000, 0x00000000, 0x80000000 } };
    return _mm_xor_ps( s, _mm_load_ps(upnpn.f) ); 
}

vectorial_inline simd4f simd4f_flip_sign_1010(const simd4f& s) {
    const _simd4f_uif unpnp = { { 0x80000000, 0x00000000, 0x80000000, 0x00000000 } };
    return _mm_xor_ps( s, _mm_load_ps(unpnp.f) ); 
}

vectorial_inline simd4f simd4f_min(const simd4f& a, const simd4f& b) {
    return _mm_min_ps( a, b ); 
}

vectorial_inline simd4f simd4f_max(const simd4f& a, const simd4f& b) {
    return _mm_max_ps( a, b ); 
}


// Bitwise

vectorial_inline simd4f simd4f_and(const simd4f& a, const simd4f& b) {
    return _mm_and_ps(a, b);
}

vectorial_inline simd4i simd4i_and(const simd4i& a, const simd4i& b) {
    return _mm_and_si128(a, b);
}


// Conditional

vectorial_inline simd4f simd4f_compare_lt(const simd4f& a, const simd4f& b)
{
    return _mm_cmplt_ps(a, b);
}


vectorial_inline simd4f simd4f_compare_gteq(const simd4f& a, const simd4f& b)
{
    return _mm_cmpge_ps(a, b);
}

vectorial_inline simd4f simd4f_compare_gt(const simd4f& a, const simd4f& b)
{
    return _mm_cmpgt_ps(a, b);
}

vectorial_inline simd4f simd4f_select(const simd4f& a, const simd4f& b, const simd4f& mask)
{
//    return _mm_or_ps(_mm_and_ps(b, mask), _mm_andnot_ps(mask, a));
    return _mm_xor_ps(a, _mm_and_ps(mask, _mm_xor_ps(b, a)));
}

vectorial_inline simd4i simd4i_select(const simd4i& a, const simd4i& b, const simd4i& mask)
{
    return _mm_xor_si128(a, _mm_and_si128(mask, _mm_xor_si128(b, a)));
}




#ifdef __cplusplus
}
#endif


#endif

