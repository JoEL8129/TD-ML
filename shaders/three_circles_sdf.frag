// GLSL TOP: three circles SDF, smooth union, edge softness, single color.
// Uniforms: u_pos0/1/2 (vec2), u_radius0/1/2 (float), u_blendFactor, u_edgeSoftness, u_color (vec3).
// Bind from CHOPs or TOP parameters; array uniforms u_pos[3]/u_radius[3] optional where supported.

out vec4 fragColor;

uniform vec2 u_pos0;
uniform vec2 u_pos1;
uniform vec2 u_pos2;
uniform float u_radius0;
uniform float u_radius1;
uniform float u_radius2;
uniform float u_blendFactor;     // smooth-union k (e.g. 0.02–0.1)
uniform float u_edgeSoftness;    // edge AA width (e.g. 0.005–0.02)
uniform vec3 u_color;            // fill RGB

// SDF: signed distance to circle (p in circle space, r radius)
float sdCircle(vec2 p, float r)
{
    return length(p) - r;
}

// Polynomial smooth min of two SDFs (k = blend width)
float smoothUnion(float d1, float d2, float k)
{
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

void main()
{
    // UVs in [-1, 1]; aspect depends on TOP resolution
    vec2 uv = vUV.st * 2.0 - 1.0;

    float d = 1e6;   // start with “far” distance
    d = smoothUnion(d, sdCircle(uv - u_pos0, u_radius0), u_blendFactor);
    d = smoothUnion(d, sdCircle(uv - u_pos1, u_radius1), u_blendFactor);
    d = smoothUnion(d, sdCircle(uv - u_pos2, u_radius2), u_blendFactor);

    // SDF -> alpha: soft edge via smoothstep(edge, -edge, d)
    float edge = u_edgeSoftness;
    float mask = smoothstep(edge, -edge, d);

    vec4 color = vec4(u_color * mask, 1.0);
    fragColor = TDOutputSwizzle(color);
}
