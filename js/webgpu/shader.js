export const wgslSource = `
struct Params {
  data : array<vec4f, 12>,
};
@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var srcSampler : sampler;
@group(0) @binding(2) var srcSamplerLinear : sampler;
@group(0) @binding(3) var srcTex : texture_2d<f32>;
@group(0) @binding(4) var histTex : texture_2d<f32>;

struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0) uv : vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vid : u32) -> VSOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0)
  );
  var out : VSOut;
  out.pos = vec4f(positions[vid], 0.0, 1.0);
  out.uv = (out.pos.xy * 0.5) + vec2f(0.5);
  return out;
}

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn saturate3(v: vec3f) -> vec3f { return clamp(v, vec3f(0.0), vec3f(1.0)); }
fn hdrEncode(c: vec3f) -> vec3f { return c / (c + vec3f(1.0)); }
fn hdrDecode(c: vec3f) -> vec3f { return c / max(vec3f(1.0) - c, vec3f(1e-4)); }
fn hash21(p: vec2f) -> f32 {
  let h = dot(p, vec2f(127.1, 311.7));
  return fract(sin(h) * 43758.5453123);
}
fn tri(x: f32) -> f32 {
  let f = fract(x);
  return 1.0 - abs(f * 2.0 - 1.0);
}
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn gauss(x: f32, sigma: f32, k: f32) -> f32 {
  let n = x / max(1e-6, sigma);
  return exp(-k * n * n);
}

fn sampleTex(src: vec2f) -> vec3f {
  let p0 = P.data[0];
  let sz = p0.xy;
  return textureSampleLevel(srcTex, srcSampler, (src + vec2f(0.5)) / sz, 0.0).rgb;
}

fn sampleTexLinear(src: vec2f) -> vec3f {
  let p0 = P.data[0];
  let sz = p0.xy;
  return textureSampleLevel(srcTex, srcSamplerLinear, (src + vec2f(0.5)) / sz, 0.0).rgb;
}

fn sampleHistory(uv: vec2f) -> vec3f {
  return textureSampleLevel(histTex, srcSampler, uv, 0.0).rgb;
}

fn warpUv(uvIn: vec2f, dstSize: vec2f, curvature: f32, time: f32, syncJitter: f32, syncJitterSpeed: f32) -> vec4f {
  var uv = uvIn * 2.0 - vec2f(1.0);
  let jitterPhase = (uvIn.y + time * max(0.05, syncJitterSpeed + 0.1)) * 120.0;
  let jitter = syncJitter * 0.015 * sin(jitterPhase);
  let wobble = syncJitter * 0.010 * sin(uvIn.y * 40.0 + time * 8.0);
  uv.x += jitter;
  uv.y += wobble;

  let aspect = dstSize.x / max(dstSize.y, 1.0);
  uv.x *= aspect;
  let zoom = 1.0 + 0.6 * curvature;
  uv /= zoom;
  let r2 = dot(uv, uv);
  uv *= (1.0 + curvature * r2);
  uv.x /= aspect;
  let uvw = uv * 0.5 + vec2f(0.5);
  let dist = length(uv);
  return vec4f(uvw, dist, aspect);
}

@fragment
fn fs_beam(in: VSOut) -> @location(0) vec4f {
  let p0 = P.data[0];
  let p1 = P.data[1];
  let p2 = P.data[2];
  let p3 = P.data[3];
  let p4 = P.data[4];
  let p5 = P.data[5];
  let p6 = P.data[6];
  let p7 = P.data[7];
  let p8 = P.data[8];
  let p9 = P.data[9];
  let p10 = P.data[10];
  let p11 = P.data[11];

  let srcSize = p0.xy;
  let dstSize = p0.zw;
  let par = max(p1.xy, vec2f(1e-6));
  let scanline = p1.z;
  let beamMin = p1.w;
  let beamMax = p2.x;
  let sharp = p2.y;
  let maskStrength = p2.z;
  let bloom = p3.x;
  let bloomSigma = p3.y;
  let halation = p3.z;
  let halationSigma = p3.w;
  let cax = p4.x;
  let cay = p4.y;
  let caEdge = p4.z;
  let energyLock = p4.w;
  let maskThickness = p5.x;
  let maskRound = p6.x;
  let chromaSpill = p6.y;
  let chromaSigma = p6.z;
  let chromaGain = p6.w;
  let blackSpill = p7.x;
  let blackSigma = p7.y;
  let curvature = p7.z;
  let phosphorResponse = clamp(p8.z, 0.0, 1.0);
  let spillBias = p8.w;
  let time = p9.x;
  let syncJitter = p9.y;
  let syncJitterSpeed = p9.z;
  let bypass = p10.z;
  let maskType = u32(p10.w + 0.5);
  let hdrGain = p11.y;
  let hdrHighlights = p11.z;

  let uvIn = vec2f(in.uv.x, 1.0 - in.uv.y);
  let historyLdr = p11.w;
  if (bypass > 0.5) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }

  let warped = warpUv(uvIn, dstSize, curvature, time, syncJitter, syncJitterSpeed);
  let uvw = warped.xy;
  if (uvw.x < 0.0 || uvw.x > 1.0 || uvw.y < 0.0 || uvw.y > 1.0) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }

  var srcCoord = uvw * srcSize;
  let lineHash = hash21(vec2f(floor(uvw.y * dstSize.y), floor(time * 120.0)));
  srcCoord.x += (lineHash - 0.5) * syncJitter * 2.0;

  let lineRate = max(1.0, srcSize.y) * 59.94;
  let scanPos = fract(time * lineRate) * srcSize.y;
  let scanDist = abs(srcCoord.y - scanPos);
  let scanWidth = mix(0.9, 0.2, scanline) * mix(1.0, 1.35, phosphorResponse);
  let scanGate = exp(-scanDist * scanDist / (scanWidth * scanWidth + 1e-4));

  let yCenter = floor(srcCoord.y) + 0.5;
  let dy = (srcCoord.y - yCenter) * par.y;
  let sigmaY = 0.5 * par.y;
  let scanW = gauss(dy, sigmaY, mix(1.0, 2.8, sharp));
  let y0 = yCenter - 1.0;
  let y1 = yCenter;
  let y2 = yCenter + 1.0;
  let wy0 = gauss((srcCoord.y - y0) * par.y, sigmaY, mix(1.0, 2.8, sharp));
  let wy1 = gauss((srcCoord.y - y1) * par.y, sigmaY, mix(1.0, 2.8, sharp));
  let wy2 = gauss((srcCoord.y - y2) * par.y, sigmaY, mix(1.0, 2.8, sharp));
  let xCenter = floor(srcCoord.x) + 0.5;
  let cCenter = sampleTex(vec2f(xCenter, yCenter));
  let lum = luma(cCenter);
  let beamSigmaBase = mix(beamMin, beamMax, saturate(lum));

  let uv = vec2f((uvw.x * 2.0 - 1.0), (uvw.y * 2.0 - 1.0));
  let caEdgeStrength = pow(clamp(length(uv), 0.0, 1.0), 1.15) * caEdge;
  let halfWidth = i32(clamp(ceil(3.0 * beamSigmaBase * par.x), 3.0, 8.0));
  var accum = vec3f(0.0);
  var wsum = 0.0;
  var LsrcAcc = 0.0;
  var Lw = 0.0;
  let fx = (srcCoord.x - xCenter) * par.x;

  for (var i: i32 = -12; i <= 12; i = i + 1) {
    if (abs(i) > halfWidth) { continue; }
    let offset = f32(i);
    let sampleX = xCenter + offset;
    let dx = (fx - offset * par.x);
    let w = gauss(dx, beamSigmaBase * par.x, mix(1.0, 3.0, sharp));

    let caAmount = vec2f(cax, cay) * par;
    let caStrength = mix(0.35, 1.35, clamp(caEdgeStrength, 0.0, 1.0));

    if (wy0 > 1e-4) {
      let base0 = vec2f(sampleX, y0);
      let lens0 = (base0 - (srcSize * 0.5)) / max(vec2f(1.0), srcSize);
      let weighted0 = vec2f(lens0.x, lens0.y * (srcSize.x / max(srcSize.y, 1.0)));
      let dirLen0 = length(weighted0);
      let caDir0 = select(vec2f(0.0), weighted0 / dirLen0, dirLen0 > 0.0);
      let pixelShift0 = caDir0 * caAmount * caStrength;
      let tapR0 = sampleTexLinear(base0 - pixelShift0).r;
      let tapG0 = sampleTexLinear(base0).g;
      let tapB0 = sampleTexLinear(base0 + pixelShift0).b;
      let s0 = vec3f(tapR0, tapG0, tapB0);
      let w0 = w * wy0;
      accum += s0 * w0;
      wsum += w0;
      LsrcAcc += luma(s0) * w0;
      Lw += w0;
    }

    if (wy1 > 1e-4) {
      let base1 = vec2f(sampleX, y1);
      let lens1 = (base1 - (srcSize * 0.5)) / max(vec2f(1.0), srcSize);
      let weighted1 = vec2f(lens1.x, lens1.y * (srcSize.x / max(srcSize.y, 1.0)));
      let dirLen1 = length(weighted1);
      let caDir1 = select(vec2f(0.0), weighted1 / dirLen1, dirLen1 > 0.0);
      let pixelShift1 = caDir1 * caAmount * caStrength;
      let tapR1 = sampleTexLinear(base1 - pixelShift1).r;
      let tapG1 = sampleTexLinear(base1).g;
      let tapB1 = sampleTexLinear(base1 + pixelShift1).b;
      let s1 = vec3f(tapR1, tapG1, tapB1);
      let w1 = w * wy1;
      accum += s1 * w1;
      wsum += w1;
      LsrcAcc += luma(s1) * w1;
      Lw += w1;
    }

    if (wy2 > 1e-4) {
      let base2 = vec2f(sampleX, y2);
      let lens2 = (base2 - (srcSize * 0.5)) / max(vec2f(1.0), srcSize);
      let weighted2 = vec2f(lens2.x, lens2.y * (srcSize.x / max(srcSize.y, 1.0)));
      let dirLen2 = length(weighted2);
      let caDir2 = select(vec2f(0.0), weighted2 / dirLen2, dirLen2 > 0.0);
      let pixelShift2 = caDir2 * caAmount * caStrength;
      let tapR2 = sampleTexLinear(base2 - pixelShift2).r;
      let tapG2 = sampleTexLinear(base2).g;
      let tapB2 = sampleTexLinear(base2 + pixelShift2).b;
      let s2 = vec3f(tapR2, tapG2, tapB2);
      let w2 = w * wy2;
      accum += s2 * w2;
      wsum += w2;
      LsrcAcc += luma(s2) * w2;
      Lw += w2;
    }
  }
  var beamColor = accum / max(wsum, 1e-5);

  let Lsrc = select(luma(cCenter), LsrcAcc / Lw, Lw > 0.0);
  let Lb = luma(beamColor) + 1e-5;
  beamColor *= mix(1.0, Lsrc / Lb, clamp(energyLock, 0.0, 1.0));

  if (bloom > 0.0) {
    let sigmaPx = max(1.2, bloomSigma * 2.2);
    let radius = i32(clamp(floor(sigmaPx), 1.0, 6.0));
    var bloomAcc = vec3f(0.0);
    var bloomW = 0.0;
    for (var oy: i32 = -6; oy <= 6; oy = oy + 1) {
      if (abs(oy) > radius) { continue; }
      for (var ox: i32 = -6; ox <= 6; ox = ox + 1) {
        if (abs(ox) > radius) { continue; }
        let offset = vec2f(f32(ox) * par.x, f32(oy) * par.y);
        let dist2 = dot(offset, offset);
        let w = exp(-dist2 / (2.0 * sigmaPx * sigmaPx + 1e-4));
        let s = sampleTex(vec2f(xCenter + f32(ox), yCenter + f32(oy)));
        bloomAcc += s * w;
        bloomW += w;
      }
    }
    var bloomCol = bloomAcc / max(bloomW, 1e-4);
    bloomCol = mix(bloomCol, vec3f(luma(bloomCol)), 0.4);
    let bloomMix = clamp(bloom, 0.0, 1.0);
    beamColor = mix(beamColor, beamColor + bloomCol, bloomMix);
  }

  if (chromaSpill > 0.0) {
    let cSigma = beamSigmaBase * chromaSigma * par.x;
    let halfC = i32(clamp(ceil(3.0 * cSigma), 4.0, 14.0));
    var cAccL = vec3f(0.0);
    var cWL = 0.0;
    var cAccR = vec3f(0.0);
    var cWR = 0.0;
    for (var t: i32 = -16; t <= 16; t = t + 1) {
      if (abs(t) > halfC) { continue; }
      let offset = f32(t);
      let sampleX = xCenter + offset;
      let s = sampleTex(vec2f(sampleX, yCenter));
      let L = luma(s);
      let chroma = s - vec3f(L);
      let dx = (fx - offset * par.x);
      let w = gauss(dx, cSigma, 0.7);
      if (offset < 0.0) {
        cAccL += chroma * w; cWL += w;
      } else {
        cAccR += chroma * w; cWR += w;
      }
    }
    let cL = select(vec3f(0.0), cAccL / cWL, cWL > 0.0);
    let cR = select(vec3f(0.0), cAccR / cWR, cWR > 0.0);
    let bias = clamp(spillBias * 0.5 + 0.5, 0.0, 1.0);
    let chromaBlur = mix(cL, cR, bias);
    let Lonly = vec3f(luma(beamColor));
    var Conly = beamColor - Lonly;
    let maskBoost = 1.0 + 0.5 * clamp(maskStrength, 0.0, 1.0);
    Conly += chromaBlur * (chromaSpill * max(0.0, chromaGain) * maskBoost);
    beamColor = Lonly + Conly;
  }

  if (halation > 0.0) {
    let hSigma = beamSigmaBase * halationSigma * par.x;
    let halfH = i32(clamp(ceil(3.0 * hSigma), 6.0, 20.0));
    var hAcc = 0.0;
    var hW = 0.0;
    for (var kx: i32 = -24; kx <= 24; kx = kx + 1) {
      if (abs(kx) > halfH) { continue; }
      let offset = f32(kx);
      let sampleX = xCenter + offset;
      let dx = (fx - offset * par.x);
      let w = gauss(dx, hSigma, 0.6);
      hAcc += luma(sampleTex(vec2f(sampleX, yCenter))) * w;
      hW += w;
    }
    let hCol = select(luma(beamColor), hAcc / hW, hW > 0.0);
    let halo = mix(vec3f(hCol), beamColor, 0.25);
    beamColor += halation * halo;
  }

  if (blackSpill > 0.0) {
    let dSigma = beamSigmaBase * blackSigma * par.x;
    let halfD = i32(clamp(ceil(3.0 * dSigma), 4.0, 14.0));
    var dAcc = 0.0;
    var dW = 0.0;
    for (var q: i32 = -16; q <= 16; q = q + 1) {
      if (abs(q) > halfD) { continue; }
      let offset = f32(q);
      let sampleX = xCenter + offset;
      let Ls = luma(sampleTex(vec2f(sampleX, yCenter)));
      let inv = 1.0 - Ls;
      let dx = (fx - offset * par.x);
      let w = gauss(dx, dSigma, 0.7);
      dAcc += inv * w;
      dW += w;
    }
    let darkBlur = select(0.0, dAcc / dW, dW > 0.0);
    let lumNow = clamp(luma(beamColor), 0.0, 1.0);
    let strength = blackSpill * 0.5 * darkBlur * mix(0.5, 1.0, lumNow);
    beamColor *= (1.0 - strength);
  }

  beamColor *= max(0.0, hdrGain);
  let preLum = luma(beamColor);
  let hdrBoost = pow(max(preLum - 0.42, 0.0) * 3.2, 1.25) * 4.6;
  beamColor += vec3f(hdrBoost * max(0.0, hdrHighlights));

  beamColor *= mix(1.0, scanW, scanline);
  beamColor *= mix(1.0, scanGate, scanline);

  if (maskStrength > 0.0) {
    let subPos = srcCoord.x * 3.0;
    let row = i32(floor(srcCoord.y)) & 1;
    let dotMask = maskType == 2u;
    let rowOffset = select(0.0, 0.5, dotMask && (row == 1));
    let triadX = subPos + rowOffset;
    let sub = i32(floor(triadX)) % 3;
    let subX = fract(triadX);
    let dx = abs(subX - 0.5) / 0.5;
    let cellY = fract(srcCoord.y);
    let dy2 = abs(cellY - 0.5) / 0.5;
    let thickness = clamp(maskThickness, 0.2, 2.0);
    let width = mix(0.12, 0.48, clamp(thickness * 0.5, 0.0, 1.0));
    let height = mix(0.25, 0.95, clamp(maskRound, 0.0, 1.0));
    var mask = vec3f(1.0);
    if (maskType == 0u) {
      let stripe = exp(-pow(dx / width, 2.0));
      let subMask = select(
        select(vec3f(1.0, 0.35, 0.35), vec3f(0.35, 1.0, 0.35), sub == 1),
        vec3f(0.35, 0.35, 1.0),
        sub == 2
      );
      mask = subMask * stripe;
    } else if (maskType == 1u) {
      let stripe = exp(-pow(dx / width, 2.0));
      mask = vec3f(stripe);
    } else {
      let spot = exp(-pow(dx / width, 2.0)) * exp(-pow(dy2 / height, 2.0));
      let subMask = select(
        select(vec3f(1.0, 0.35, 0.35), vec3f(0.35, 1.0, 0.35), sub == 1),
        vec3f(0.35, 0.35, 1.0),
        sub == 2
      );
      mask = subMask * spot;
    }
    let grain = 0.92 + 0.12 * hash21(vec2f(floor(triadX), floor(srcCoord.y)));
    mask *= grain;
    beamColor *= mix(vec3f(1.0), mask, maskStrength);
  }

  let postLum = luma(beamColor);
  let hot2 = pow(max(postLum - 0.30, 0.0), 1.2) * 1.6 * max(0.0, hdrHighlights);
  beamColor += vec3f(hot2);

  let phosphorK = mix(0.9, 3.2, phosphorResponse) * (1.0 + 0.6 * max(0.0, hdrHighlights));
  let phosphorMix = mix(0.15, 0.75, phosphorResponse);
  let phosphor = vec3f(1.0) - exp(-beamColor * phosphorK);
  beamColor = mix(beamColor, phosphor * (1.05 + 0.35 * max(0.0, hdrHighlights)), phosphorMix);

  var outColor = max(beamColor, vec3f(0.0));
  if (historyLdr > 0.5) {
    outColor = hdrEncode(outColor);
  }
  return vec4f(outColor, 1.0);
}

@fragment
fn fs_final(in: VSOut) -> @location(0) vec4f {
  let p0 = P.data[0];
  let p2 = P.data[2];
  let p7 = P.data[7];
  let p8 = P.data[8];
  let p9 = P.data[9];
  let p10 = P.data[10];
  let p11 = P.data[11];

  let dstSize = p0.zw;
  let gamma = p2.w;
  let curvature = p7.z;
  let vignette = p7.w;
  let postContrast = p8.x;
  let postSaturation = p8.y;
  let time = p9.x;
  let syncJitter = p9.y;
  let syncJitterSpeed = p9.z;
  let rollingBand = p9.w;
  let rollingBandSpeed = p10.x;
  let noiseStrength = p10.y;
  let bypass = p10.z;
  let debugHDR = p11.x;
  let historyLdr = p11.w;

  if (bypass > 0.5) {
    let src = textureSampleLevel(srcTex, srcSamplerLinear, in.uv, 0.0).rgb;
    let col = pow(saturate3(src), vec3f(1.0 / gamma));
    return vec4f(col, 1.0);
  }

  let uvIn = vec2f(in.uv.x, 1.0 - in.uv.y);
  let warped = warpUv(uvIn, dstSize, curvature, time, syncJitter, syncJitterSpeed);
  let uvw = warped.xy;
  let dist = warped.z;
  if (uvw.x < 0.0 || uvw.x > 1.0 || uvw.y < 0.0 || uvw.y > 1.0) {
    let bezel = vec3f(0.015, 0.015, 0.02);
    return vec4f(bezel, 1.0);
  }

  let histEncoded = sampleHistory(uvIn);
  var col = histEncoded;
  if (historyLdr > 0.5) {
    col = hdrDecode(histEncoded);
  }

  let vig = 1.0 - vignette * smoothstep(0.6, 1.0, dist);
  col *= vig;

  if (rollingBand > 0.0) {
    let speed = rollingBandSpeed;
    let center = fract(time * speed);
    let bandDist = abs(fract(uvw.y - center + 0.5) - 0.5);
    let bandWidth = mix(0.03, 0.16, clamp(rollingBand, 0.0, 1.0));
    let bandMask = exp(-bandDist * bandDist / (bandWidth * bandWidth + 1e-4));
    let bandPulse = 0.5 + 0.5 * sin((uvw.x * dstSize.x * 0.5) + time * 12.0);
    col = mix(col, col * (0.7 + 0.3 * bandPulse), bandMask * rollingBand);
  }

  if (noiseStrength > 0.0) {
    let noiseCoord = vec2f(uvw.x * dstSize.x, uvw.y * dstSize.y);
    let staticGrain = hash21(noiseCoord + vec2f(time * 60.0, time * 13.0)) - 0.5;
    let lineNoise = hash21(vec2f(0.0, floor(noiseCoord.y + time * 12.0))) - 0.5;
    let grain = staticGrain * 0.7 + lineNoise * 0.3;
    col += col * grain * noiseStrength;
    col = clamp(col, vec3f(0.0), vec3f(6.0));
  }

  col = (col - vec3f(0.5)) * postContrast + vec3f(0.5);
  col = max(col, vec3f(0.0));
  let Y = luma(col);
  let C = col - vec3f(Y);
  col = vec3f(Y) + C * postSaturation;

  if (debugHDR > 0.5) {
    if (uvw.y < 0.05) {
      let bar = select(vec3f(1.0), vec3f(4.0), uvw.x > 0.5);
      col = mix(col, bar, 0.8);
    }
  }

  col = pow(max(col, vec3f(0.0)), vec3f(1.0 / gamma));
  return vec4f(col, 1.0);
}
`;
