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
  let v0 = P.data[0];
  let sz = v0.xy;
  return textureSampleLevel(srcTex, srcSampler, (src + vec2f(0.5)) / sz, 0.0).rgb;
}

fn sampleTexLinear(src: vec2f) -> vec3f {
  let v0 = P.data[0];
  let sz = v0.xy;
  return textureSampleLevel(srcTex, srcSamplerLinear, (src + vec2f(0.5)) / sz, 0.0).rgb;
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
  let v0 = P.data[0];
  let v1 = P.data[1];
  let v2 = P.data[2];
  let v3 = P.data[3];
  let v4 = P.data[4];
  let v5 = P.data[5];
  let v6 = P.data[6];
  let v7 = P.data[7];
  let v8 = P.data[8];
  let v9 = P.data[9];
  let v10 = P.data[10];
  let v11 = P.data[11];

  let srcSize = v0.xy;
  let dstSize = v0.zw;
  let par = max(v1.xy, vec2f(1e-6));
  let beamMin = v1.w;
  let beamMax = v2.x;
  let sharp = v2.y;
  let maskStrength = v2.z;
  let bloom = v3.x;
  let bloomSigma = v3.y;
  let halation = v3.z;
  let halationSigma = v3.w;
  let cax = v4.x;
  let cay = v4.y;
  let caEdge = v4.z;
  let energyLock = v4.w;
  let decay = vec3f(v5.y, v5.z, v5.w);
  let chromaSpill = v6.y;
  let chromaSigma = v6.z;
  let chromaGain = v6.w;
  let blackSpill = v7.x;
  let blackSigma = v7.y;
  let curvature = v7.z;
  let spillBias = v8.w;
  let time = v9.x;
  let syncJitter = v9.y;
  let syncJitterSpeed = v9.z;
  let bypass = v10.z;
  let hdrGain = v11.y;
  let hdrHighlights = v11.z;

  let prev = textureSampleLevel(histTex, srcSamplerLinear, in.uv, 0.0).rgb;
  if (bypass > 0.5) {
    return vec4f(prev, 1.0);
  }

  let warped = warpUv(in.uv, dstSize, curvature, time, syncJitter, syncJitterSpeed);
  let uvw = warped.xy;
  if (uvw.x < 0.0 || uvw.x > 1.0 || uvw.y < 0.0 || uvw.y > 1.0) {
    let faded = prev * decay;
    return vec4f(faded, 1.0);
  }

  let srcCoord = uvw * srcSize;
  let lineHash = hash21(vec2f(floor(uvw.y * dstSize.y), floor(time * 120.0)));
  srcCoord.x += (lineHash - 0.5) * syncJitter * 2.0;

  let yCenter = floor(srcCoord.y) + 0.5;
  let xCenter = floor(srcCoord.x) + 0.5;
  let cCenter = sampleTex(vec2f(xCenter, yCenter));
  let lum = luma(cCenter);
  let beamSigmaBase = mix(beamMin, beamMax, saturate(lum));

  let uv = vec2f((uvw.x * 2.0 - 1.0), (uvw.y * 2.0 - 1.0));
  let caEdgeStrength = pow(clamp(length(uv), 0.0, 1.0), 1.15) * caEdge;
  let halfWidth = i32(clamp(ceil(3.0 * beamSigmaBase * par.x), 3.0, 8.0));
  var accum = vec3f(0.0);
  var wsum = 0.0;
  let fx = (srcCoord.x - xCenter) * par.x;

  for (var i: i32 = -12; i <= 12; i = i + 1) {
    if (abs(i) > halfWidth) { continue; }
    let offset = f32(i);
    let sampleX = xCenter + offset;
    let base = vec2f(sampleX, yCenter);
    let lensNorm = (base - (srcSize * 0.5)) / max(vec2f(1.0), srcSize);
    let weighted = vec2f(lensNorm.x, lensNorm.y * (srcSize.x / max(srcSize.y, 1.0)));
    let dirLen = length(weighted);
    let caDir = select(vec2f(0.0), weighted / dirLen, dirLen > 0.0);
    let caAmount = vec2f(cax, cay) * par;
    let caStrength = mix(0.35, 1.35, clamp(caEdgeStrength, 0.0, 1.0));
    let pixelShift = caDir * caAmount * caStrength;
    let tapR = sampleTexLinear(base - pixelShift).r;
    let tapG = sampleTexLinear(base).g;
    let tapB = sampleTexLinear(base + pixelShift).b;
    let s = vec3f(tapR, tapG, tapB);
    let dx = (fx - offset * par.x);
    let w = gauss(dx, beamSigmaBase * par.x, mix(1.0, 3.0, sharp));
    accum += s * w;
    wsum += w;
  }
  var beamColor = accum / max(wsum, 1e-5);

  var LsrcAcc = 0.0;
  var Lw = 0.0;
  for (var j: i32 = -12; j <= 12; j = j + 1) {
    if (abs(j) > halfWidth) { continue; }
    let offset = f32(j);
    let sampleX = xCenter + offset;
    let s = sampleTex(vec2f(sampleX, yCenter));
    let dx = (fx - offset * par.x);
    let w = gauss(dx, beamSigmaBase * par.x, mix(1.0, 3.0, sharp));
    LsrcAcc += luma(s) * w;
    Lw += w;
  }
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
  let hdrBoost = pow(max(preLum - 0.6, 0.0) * 2.2, 2.0) * 3.0;
  beamColor += vec3f(hdrBoost * max(0.0, hdrHighlights));

  let mixed = prev * decay + beamColor * (vec3f(1.0) - decay);
  return vec4f(max(mixed, vec3f(0.0)), 1.0);
}

@fragment
fn fs_final(in: VSOut) -> @location(0) vec4f {
  let v0 = P.data[0];
  let v1 = P.data[1];
  let v2 = P.data[2];
  let v6 = P.data[6];
  let v7 = P.data[7];
  let v8 = P.data[8];
  let v9 = P.data[9];
  let v10 = P.data[10];
  let v11 = P.data[11];

  let srcSize = v0.xy;
  let dstSize = v0.zw;
  let par = max(v1.xy, vec2f(1e-6));
  let scanline = v1.z;
  let sharp = v2.y;
  let maskStrength = v2.z;
  let gamma = v2.w;
  let maskRound = v6.x;
  let curvature = v7.z;
  let vignette = v7.w;
  let postContrast = v8.x;
  let postSaturation = v8.y;
  let postBrightness = v8.z;
  let time = v9.x;
  let syncJitter = v9.y;
  let syncJitterSpeed = v9.z;
  let rollingBand = v9.w;
  let rollingBandSpeed = v10.x;
  let noiseStrength = v10.y;
  let bypass = v10.z;
  let maskType = u32(v10.w + 0.5);
  let debugHDR = v11.x;
  let maskThickness = P.data[5].x;

  if (bypass > 0.5) {
    let src = textureSampleLevel(srcTex, srcSamplerLinear, in.uv, 0.0).rgb;
    let col = pow(saturate3(src), vec3f(1.0 / gamma));
    return vec4f(col, 1.0);
  }

  let warped = warpUv(in.uv, dstSize, curvature, time, syncJitter, syncJitterSpeed);
  let uvw = warped.xy;
  let dist = warped.z;
  if (uvw.x < 0.0 || uvw.x > 1.0 || uvw.y < 0.0 || uvw.y > 1.0) {
    let bezel = vec3f(0.015, 0.015, 0.02);
    return vec4f(bezel, 1.0);
  }

  var col = textureSampleLevel(histTex, srcSamplerLinear, in.uv, 0.0).rgb;

  let srcCoord = uvw * srcSize;
  let yCenter = floor(srcCoord.y) + 0.5;
  let dy = (srcCoord.y - yCenter) * par.y;
  let sigmaY = 0.5 * par.y;
  let scanW = gauss(dy, sigmaY, mix(1.0, 2.8, sharp));
  col *= mix(1.0, scanW, 1.0 - scanline);

  if (maskStrength > 0.0) {
    let cell = fract(srcCoord);
    let sub = i32(floor(cell.x * 3.0));
    let centerX = (f32(sub) + 0.5) / 3.0;
    let dx = abs(cell.x - centerX) / (1.0 / 6.0);
    let dy2 = abs(cell.y - 0.5) / 0.5;
    let thickness = clamp(maskThickness, 0.2, 2.0);
    let width = mix(0.4, 1.15, clamp(thickness * 0.5, 0.0, 1.0));
    let roundness = mix(0.6, 2.0, clamp(maskRound, 0.0, 1.0));
    var mask = vec3f(1.0);
    if (maskType == 0u) {
      let spot = exp(-pow(dx / width, 2.0)) * exp(-pow(dy2 * roundness, 2.0));
      let subMask = select(
        select(vec3f(1.0, 0.35, 0.35), vec3f(0.35, 1.0, 0.35), sub == 1),
        vec3f(0.35, 0.35, 1.0),
        sub == 2
      );
      mask = subMask * spot;
    } else {
      let stripe = exp(-pow(dx / width, 2.0));
      let vert = exp(-pow(dy2 * (0.6 + roundness * 0.4), 2.0));
      mask = vec3f(stripe * mix(1.0, vert, maskRound));
    }
    col *= mix(vec3f(1.0), mask, maskStrength);
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

  col = max(col + vec3f(postBrightness), vec3f(0.0));
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
