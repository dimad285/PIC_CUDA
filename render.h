#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <kernels.cuh>
#include <chrono>
#include <vector>

GLuint g_vbo;
GLint g_colorLoc, g_pointSizeLoc;
struct cudaGraphicsResource* g_cudaVboResource;
GLFWwindow* g_window = nullptr;

// Line segment rendering resources
GLuint g_lineVBO;
GLuint g_lineVAO;
GLuint g_lineShaderProgram;
GLint g_lineColorLoc;
std::vector<float> g_lineVertices;

// --- Constants ---
const int WINDOW_WIDTH = 1024;
const int WINDOW_HEIGHT = 768;
const char* WINDOW_TITLE = "Particle Visualization";

const char* VERTEX_SHADER_SOURCE = R"(
#version 400 core
layout(location = 0) in vec2 position;
uniform float pointSize;
uniform mat4 projection;
void main() {
    // Transform from [0,1] domain to [-1,1] NDC
    vec2 ndc = position * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = pointSize;
}
)";

const char* FRAGMENT_SHADER_SOURCE = R"(
#version 400 core
out vec4 fragColor;
uniform vec4 color;
void main() {
    // Create circular points
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (length(coord) > 0.5) {
        discard;
    }
    fragColor = color;
}
)";

// Line segment shaders
const char* LINE_VERTEX_SHADER_SOURCE = R"(
#version 400 core
layout(location = 0) in vec2 position;
void main() {
    // Transform from [0,1] domain to [-1,1] NDC
    vec2 ndc = position * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

const char* LINE_FRAGMENT_SHADER_SOURCE = R"(
#version 400 core
out vec4 fragColor;
uniform vec4 color;
void main() {
    fragColor = color;
}
)";

#define CHECK_CUDA_ERROR(err) \
    do { \
        cudaError_t error = (err); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_GL_ERROR() \
    do { \
        GLenum err; \
        while ((err = glGetError()) != GL_NO_ERROR) { \
            std::cerr << "OpenGL Error: " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while (0)

struct LineSegment {
    float x1, y1, x2, y2;
    
    LineSegment(float x1, float y1, float x2, float y2) : x1(x1), y1(y1), x2(x2), y2(y2) {}
};

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation error (Type: " << type << "):\n" << infoLog << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    CHECK_GL_ERROR();
    return shader;
}

GLuint createShaderProgram() {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, VERTEX_SHADER_SOURCE);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE);

    if (vertexShader == 0 || fragmentShader == 0) {
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader program linking error:\n" << infoLog << std::endl;
        glDeleteProgram(program);
        program = 0;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    CHECK_GL_ERROR();
    return program;
}

GLuint createLineShaderProgram() {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, LINE_VERTEX_SHADER_SOURCE);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, LINE_FRAGMENT_SHADER_SOURCE);

    if (vertexShader == 0 || fragmentShader == 0) {
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Line shader program linking error:\n" << infoLog << std::endl;
        glDeleteProgram(program);
        program = 0;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    CHECK_GL_ERROR();
    return program;
}

GLuint setupParticleRendering(int maxParticles, GLuint shaderProgram) {
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    // Allocate buffer for float2 positions (converted from double2)
    glBufferData(GL_ARRAY_BUFFER, maxParticles * sizeof(float) * 2, nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    // Register VBO with CUDA
    CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&g_cudaVboResource, g_vbo,
        cudaGraphicsMapFlagsWriteDiscard));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    g_colorLoc = glGetUniformLocation(shaderProgram, "color");
    g_pointSizeLoc = glGetUniformLocation(shaderProgram, "pointSize");

    CHECK_GL_ERROR();
    return vao;
}

void setupLineRendering() {
    // Create line shader program
    g_lineShaderProgram = createLineShaderProgram();
    if (g_lineShaderProgram == 0) {
        std::cerr << "Failed to create line shader program" << std::endl;
        return;
    }

    // Create VAO and VBO for line segments
    glGenVertexArrays(1, &g_lineVAO);
    glBindVertexArray(g_lineVAO);

    glGenBuffers(1, &g_lineVBO);
    glBindBuffer(GL_ARRAY_BUFFER, g_lineVBO);

    // Setup vertex attributes for line segments
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Get uniform locations
    g_lineColorLoc = glGetUniformLocation(g_lineShaderProgram, "color");

    CHECK_GL_ERROR();
}

void setLineSegments(const std::vector<LineSegment>& segments) {
    g_lineVertices.clear();
    
    // Convert line segments to vertex array (2 vertices per segment)
    for (const auto& segment : segments) {
        // First vertex of line
        g_lineVertices.push_back(segment.x1);
        g_lineVertices.push_back(segment.y1);
        // Second vertex of line
        g_lineVertices.push_back(segment.x2);
        g_lineVertices.push_back(segment.y2);
    }
}

void updateLineBuffer() {
    if (g_lineVertices.empty()) {
        // Clear the buffer if no vertices
        glBindBuffer(GL_ARRAY_BUFFER, g_lineVBO);
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        return;
    }

    size_t bufferSize = g_lineVertices.size() * sizeof(float);
    if (bufferSize > 0) {
        glBindBuffer(GL_ARRAY_BUFFER, g_lineVBO);
        glBufferData(GL_ARRAY_BUFFER, bufferSize, g_lineVertices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        CHECK_GL_ERROR();
    }
}

void renderLineSegments(float r = 1.0f, float g = 1.0f, float b = 1.0f, float a = 1.0f) {
    if (g_lineVertices.empty()) return;
    
    // Check if we have complete line segments (pairs of vertices)
    if (g_lineVertices.size() % 4 != 0) {
        std::cerr << "Warning: Incomplete line segments - vertex count not divisible by 4" << std::endl;
        return;
    }

    // Validate shader program and VAO
    if (g_lineShaderProgram == 0) {
        std::cerr << "Error: Line shader program not initialized" << std::endl;
        return;
    }
    
    if (g_lineVAO == 0) {
        std::cerr << "Error: Line VAO not initialized" << std::endl;
        return;
    }

    glUseProgram(g_lineShaderProgram);
    CHECK_GL_ERROR();
    
    glBindVertexArray(g_lineVAO);
    CHECK_GL_ERROR();

    // Set line properties - clamp line width to valid range
    //float clampedLineWidth = std::max(0.5f, std::min(lineWidth, 10.0f));
    glLineWidth(1.0f);
    CHECK_GL_ERROR();
    
    // Validate uniform location
    if (g_lineColorLoc != -1) {
        glUniform4f(g_lineColorLoc, r, g, b, a);
        CHECK_GL_ERROR();
    }

    // Enable line smoothing for better appearance
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    CHECK_GL_ERROR();

    // Draw all line segments (2 vertices per line segment)
    int numVertices = g_lineVertices.size() / 2;
    if (numVertices > 0 && numVertices % 2 == 0) {
        glDrawArrays(GL_LINES, 0, numVertices);
        CHECK_GL_ERROR();
    }

    glDisable(GL_BLEND);
    glDisable(GL_LINE_SMOOTH);
    glBindVertexArray(0);
    glUseProgram(0);
}

// Convenience function to add a single line segment
void addLineSegment(float x1, float y1, float x2, float y2) {
    g_lineVertices.push_back(x1);
    g_lineVertices.push_back(y1);
    g_lineVertices.push_back(x2);
    g_lineVertices.push_back(y2);
}

// Convenience function to clear all line segments
void clearLineSegments() {
    g_lineVertices.clear();
}

bool initializeGLFW_GLEW() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    g_window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, nullptr, nullptr);
    if (!g_window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(g_window);
    glfwSwapInterval(1); // Enable vsync

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return false;
    }

    while (glGetError() != GL_NO_ERROR); // Clear GLEW errors
    CHECK_GL_ERROR();
    return true;
}

void renderParticles(GLuint shaderProgram, GLuint vao, int numParticles) {
    glUseProgram(shaderProgram);

    // Set uniforms
    glUniform4f(g_colorLoc, 0.8f, 0.9f, 1.0f, 1.0f); // Light blue color
    glUniform1f(g_pointSizeLoc, 3.0f); // Larger point size for visibility

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, numParticles);
    glBindVertexArray(0);

    glDisable(GL_BLEND);
    glUseProgram(0);
    CHECK_GL_ERROR();
}

double updateVBO(Particles2D& particles) {
    auto start = std::chrono::high_resolution_clock::now();

    // Map the VBO resource for writing
    size_t num_bytes;
    float2* d_vbo_ptr;

    cudaGraphicsMapResources(1, &g_cudaVboResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes, g_cudaVboResource);

    // Convert double2 positions to float2 for OpenGL
    int threadsPerBlock = 256;
    int blocks = (particles.last_alive + threadsPerBlock - 1) / threadsPerBlock;
    convertPositionsToFloat << <blocks, threadsPerBlock >> > (particles.r, d_vbo_ptr, particles.last_alive);

    //cudaGetLastError();
    //cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &g_cudaVboResource, 0);

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

void cleanupGraphics(GLuint shaderProgram, GLuint particleVAO) {
    if (g_cudaVboResource) {
        cudaGraphicsUnregisterResource(g_cudaVboResource);
    }
    if (g_vbo) {
        glDeleteBuffers(1, &g_vbo);
    }
    if (g_lineVBO) {
        glDeleteBuffers(1, &g_lineVBO);
    }
    if (particleVAO) {
        glDeleteVertexArrays(1, &particleVAO);
    }
    if (g_lineVAO) {
        glDeleteVertexArrays(1, &g_lineVAO);
    }
    if (shaderProgram) {
        glDeleteProgram(shaderProgram);
    }
    if (g_lineShaderProgram) {
        glDeleteProgram(g_lineShaderProgram);
    }
    if (g_window) {
        glfwDestroyWindow(g_window);
    }
    glfwTerminate();
    std::cout << "Graphics resources cleaned up." << std::endl;
}