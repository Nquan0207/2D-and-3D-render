#version 330

in vec3 colorInterp;
in vec2 uTexture;
uniform sampler2D diffuse_tex;
uniform float texture_mix;
out vec4 fragColor;

void main()
{
    vec3 texColor = texture(diffuse_tex, uTexture).rgb;
    vec3 base = mix(colorInterp, texColor, texture_mix);
    fragColor = vec4(base, 1.0);
}