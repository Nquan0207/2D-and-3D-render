#version 330 core

// uniform base color controlled by UI
uniform vec3 flat_color;

// output fragment color for OpenGL
out vec4 out_color;

void main() {
    out_color = vec4(flat_color, 1.0);
}
