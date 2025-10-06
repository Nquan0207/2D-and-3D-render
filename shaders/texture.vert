#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 utexture;

uniform mat4 projection, modelview, normalMat;

out vec3 colorInterp;
out vec2 uTexture;

void main()
{
    colorInterp = color;
    vec4 vp = modelview * vec4(position, 1.0);
    uTexture = vec2(utexture.x, 1.0 - utexture.y);
    gl_Position = projection * vp;
}
