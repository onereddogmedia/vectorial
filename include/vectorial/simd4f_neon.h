/*
  Vectorial
  Copyright (c) 2010 Mikko Lehtonen
  Licensed under the terms of the two-clause BSD License (see LICENSE)
*/
#ifndef VECTORIAL_SIMD4F_NEON_H
#define VECTORIAL_SIMD4F_NEON_H

#include <arm_neon.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef float32x4_t simd4f;
typedef int32x4_t simd4i;

typedef union {
    simd4f s ;
    float f[4];
} _simd4f_union;



vectorial_inline simd4f simd4f_create(const float& x, const float& y, const float& z, const float& w) {
    const float32_t d[4] = { x,y,z,w };
    simd4f s = vld1q_f32(d);
    return s;
}

vectorial_inline simd4f simd4i_create(const int& x, const int& y, const int& z, const int& w) {
    const int d[4] = { x,y,z,w };
    simd4i s = vld1q_s32(d);
    return s;
}

vectorial_inline simd4f simd4f_zero() { return vdupq_n_f32(0.0f); }

vectorial_inline simd4i simd4i_zero() { return vdupq_n_s32(0); }

vectorial_inline simd4f simd4f_uload4(const float *ary) {
    const float32_t* ary32 = (const float32_t*)ary;
    simd4f s = vld1q_f32(ary32);    
    return s;
}

vectorial_inline simd4f simd4f_uload3(const float *ary) {
    simd4f s = simd4f_create(ary[0], ary[1], ary[2], 0);
    return s;
}

vectorial_inline simd4f simd4f_uload2(const float *ary) {
    const float32_t* ary32 = (const float32_t*)ary;
    float32x2_t low = vld1_f32(ary32);
    const float32_t zero = 0;
    float32x2_t high = vld1_dup_f32(&zero); // { 0,0 } but stupid warnings from llvm-gcc
    return vcombine_f32(low, high);
}


vectorial_inline void simd4f_ustore4(const simd4f& val, float *ary) {
    vst1q_f32( (float32_t*)ary, val);
}

vectorial_inline void simd4f_ustore3(const simd4f& val, float *ary) {
    _simd4f_union u = { val };
    ary[0] = u.f[0];
    ary[1] = u.f[1];
    ary[2] = u.f[2];
}

vectorial_inline void simd4f_ustore2(const simd4f& val, float *ary) {
    const float32x2_t low = vget_low_f32(val);
    vst1_f32( (float32_t*)ary, low);
}




vectorial_inline simd4f simd4f_splat(float v) { 
    simd4f s = vdupq_n_f32(v);
    return s;
}

// todo: or is simd4f_splat(simd4f_get_x(v))  better?

vectorial_inline simd4f simd4f_splat_x(const simd4f& v) {
    float32x2_t o = vget_low_f32(v);
    simd4f ret = vdupq_lane_f32(o, 0);
    return ret;
}

vectorial_inline simd4f simd4f_splat_y(const simd4f& v) { 
    float32x2_t o = vget_low_f32(v);
    simd4f ret = vdupq_lane_f32(o, 1);
    return ret;
}

vectorial_inline simd4f simd4f_splat_z(const simd4f& v) { 
    float32x2_t o = vget_high_f32(v);
    simd4f ret = vdupq_lane_f32(o, 0);
    return ret;
}

vectorial_inline simd4f simd4f_splat_w(const simd4f& v) { 
    float32x2_t o = vget_high_f32(v);
    simd4f ret = vdupq_lane_f32(o, 1);
    return ret;
}

vectorial_inline simd4i simd4i_splat(const int& v) { 
    simd4i s = vdupq_n_s32(v);
    return s;
}

vectorial_inline simd4i simd4f_convert(const simd4f& v) { 
    simd4i s = vcvtq_s32_f32(v);
    return s;
}

vectorial_inline simd4f simd4i_convert(const simd4i& v) { 
    simd4i s = vcvtq_f32_s32(v);
    return s;
}

vectorial_inline simd4f simd4f_reciprocal(const simd4f& v) { 
    simd4f estimate = vrecpeq_f32(v);
    estimate = vmulq_f32(vrecpsq_f32(estimate, v), estimate);
    estimate = vmulq_f32(vrecpsq_f32(estimate, v), estimate);
    return estimate;
}

vectorial_inline simd4f simd4f_rsqrt(const simd4f& v) { 
    simd4f estimate = vrsqrteq_f32(v);
    simd4f estimate2 = vmulq_f32(estimate, v);
    estimate = vmulq_f32(estimate, vrsqrtsq_f32(estimate2, estimate));
    estimate2 = vmulq_f32(estimate, v);
    estimate = vmulq_f32(estimate, vrsqrtsq_f32(estimate2, estimate));
    estimate2 = vmulq_f32(estimate, v);
    estimate = vmulq_f32(estimate, vrsqrtsq_f32(estimate2, estimate));
    return estimate;
}

vectorial_inline simd4f simd4f_sqrt(const simd4f& v) { 
    return simd4f_reciprocal(simd4f_rsqrt(v));
}



// arithmetics

vectorial_inline simd4f simd4f_add(const simd4f& lhs, const simd4f& rhs) {
    simd4f ret = vaddq_f32(lhs, rhs);
    return ret;
}

vectorial_inline simd4f simd4f_sub(const simd4f& lhs, const simd4f& rhs) {
    simd4f ret = vsubq_f32(lhs, rhs);
    return ret;
}

vectorial_inline simd4f simd4f_mul(const simd4f& lhs, const simd4f& rhs) {
    simd4f ret = vmulq_f32(lhs, rhs);
    return ret;
}

vectorial_inline simd4f simd4f_div(const simd4f& lhs, const simd4f& rhs) {
    simd4f recip = simd4f_reciprocal( rhs );
    simd4f ret = vmulq_f32(lhs, recip);
    return ret;
}

vectorial_inline simd4f simd4f_madd(const simd4f& m1, const simd4f& m2, const simd4f& a) {
    return vmlaq_f32( a, m1, m2 );
}

vectorial_inline simd4i simd4i_add(const simd4i& lhs, const simd4i& rhs) {
    simd4i ret = vaddq_s32(lhs, rhs);
    return ret;
}

vectorial_inline simd4f simd4f_neg(const simd4f& v) { 
    simd4i ret = vnegq_f32(v);
    return ret;
}


vectorial_inline float simd4f_get_x(const simd4f& s) { return vgetq_lane_f32(s, 0); }
vectorial_inline float simd4f_get_y(const simd4f& s) { return vgetq_lane_f32(s, 1); }
vectorial_inline float simd4f_get_z(const simd4f& s) { return vgetq_lane_f32(s, 2); }
vectorial_inline float simd4f_get_w(const simd4f& s) { return vgetq_lane_f32(s, 3); }

vectorial_inline int simd4i_get_x(const simd4i& s) { return vgetq_lane_s32(s, 0); }
vectorial_inline int simd4i_get_y(const simd4i& s) { return vgetq_lane_s32(s, 1); }
vectorial_inline int simd4i_get_z(const simd4i& s) { return vgetq_lane_s32(s, 2); }
vectorial_inline int simd4i_get_w(const simd4i& s) { return vgetq_lane_s32(s, 3); }

vectorial_inline simd4f simd4f_cross3(const simd4f& lhs, const simd4f& rhs) {
    
    const simd4f lyzx = simd4f_create(simd4f_get_y(lhs), simd4f_get_z(lhs), simd4f_get_x(lhs), simd4f_get_w(lhs));
    const simd4f lzxy = simd4f_create(simd4f_get_z(lhs), simd4f_get_x(lhs), simd4f_get_y(lhs), simd4f_get_w(lhs));

    const simd4f ryzx = simd4f_create(simd4f_get_y(rhs), simd4f_get_z(rhs), simd4f_get_x(rhs), simd4f_get_w(rhs));
    const simd4f rzxy = simd4f_create(simd4f_get_z(rhs), simd4f_get_x(rhs), simd4f_get_y(rhs), simd4f_get_w(rhs));

    return vmlsq_f32(vmulq_f32(lyzx, rzxy), lzxy, ryzx);

}

vectorial_inline simd4f simd4f_shuffle_wxyz(const simd4f& s) { 
    _simd4f_union u = {s};
    return simd4f_create( u.f[3], u.f[0], u.f[1], u.f[2]); 
}

vectorial_inline simd4f simd4f_shuffle_zwxy(const simd4f& s) { 
    _simd4f_union u = {s};
    return simd4f_create(u.f[2], u.f[3], u.f[0], u.f[1]); 
}

vectorial_inline simd4f simd4f_shuffle_yzwx(const simd4f& s) { 
    _simd4f_union u = {s};
    return simd4f_create(u.f[1], u.f[2], u.f[3], u.f[0]); 
}

vectorial_inline simd4f simd4f_zero_w(simd4f s) {
    _simd4f_union u = {s};
    return simd4f_create(u.f[0], u.f[1], u.f[2], 0.0f);
}

vectorial_inline simd4f simd4f_zero_zw(simd4f s) {
    _simd4f_union u = {s};
    return simd4f_create(u.f[0], u.f[1], 0.0f, 0.0f);
}

vectorial_inline simd4f simd4f_merge_high(const simd4f& xyzw, const simd4f& abcd) { 
    _simd4f_union u1 = {xyzw};
    _simd4f_union u2 = {abcd};
    return simd4f_create(u1.f[2], u1.f[3], u2.f[2], u2.f[3]); 
}

vectorial_inline simd4f simd4f_flip_sign_0101(const simd4f& s) {
    const unsigned int upnpn[4] = { 0x00000000, 0x80000000, 0x00000000, 0x80000000 };
    const uint32x4_t pnpn = vld1q_u32( upnpn );
    return vreinterpretq_f32_u32( veorq_u32( vreinterpretq_u32_f32(s), pnpn ) ); 
}

vectorial_inline simd4f simd4f_flip_sign_1010(const simd4f& s) {
    const unsigned int unpnp[4] = { 0x80000000, 0x00000000, 0x80000000, 0x00000000 };
    const uint32x4_t npnp = vld1q_u32( unpnp );
    return vreinterpretq_f32_u32( veorq_u32( vreinterpretq_u32_f32(s), npnp ) ); 
}


vectorial_inline simd4f simd4f_min(const simd4f& a, const simd4f& b) {
    return vminq_f32( a, b ); 
}

vectorial_inline simd4f simd4f_max(const simd4f& a, const simd4f& b) {
    return vmaxq_f32( a, b ); 
}


// Bitwise

vectorial_inline simd4f simd4f_and(const simd4f& a, const simd4f& b) {
    return vandq_s32(a, b);
}

vectorial_inline simd4i simd4i_and(const simd4i& a, const simd4i& b) {
    return vandq_s32(a, b);
}


// Conditional

vectorial_inline simd4f simd4f_compare_lt(const simd4f& a, const simd4f& b)
{
    return vcltq_f32(a, b);
}

vectorial_inline simd4f simd4f_compare_gteq(const simd4f& a, const simd4f& b)
{
    return vcgeq_f32(a, b);
}

vectorial_inline simd4f simd4f_compare_gt(const simd4f& a, const simd4f& b)
{
    return vcgtq_f32(a, b);
}

vectorial_inline simd4f simd4f_select(const simd4f& a, const simd4f& b, const simd4f& mask)
{
    return vbslq_f32(mask, b, a);
}

vectorial_inline simd4i simd4i_select(const simd4i& a, const simd4i& b, const simd4i& mask)
{
    return vbslq_s32(mask, b, a);
}


#ifdef __cplusplus
}
#endif


#endif

