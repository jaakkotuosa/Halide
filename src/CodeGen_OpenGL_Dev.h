#ifndef HALIDE_CODEGEN_OPENGL_DEV_H
#define HALIDE_CODEGEN_OPENGL_DEV_H

/** \file
 * Defines the code-generator for producing GLSL kernel code
 */

#include "CodeGen_C.h"
#include "CodeGen_GPU_Dev.h"
#include "Target.h"

#include <sstream>
#include <map>
#include <string>

namespace Halide {
namespace Internal {

class CodeGen_GLSL;

class CodeGen_OpenGL_Dev : public CodeGen_GPU_Dev {
public:
    CodeGen_OpenGL_Dev(const Target &target);
    ~CodeGen_OpenGL_Dev();

    // CodeGen_GPU_Dev interface
    void add_kernel(Stmt stmt, const std::string &name,
                    const std::vector<GPU_Argument> &args);
    void init_module();
    std::vector<char> compile_to_src();
    std::string get_current_kernel_name();
    void dump();

private:
    CodeGen_GLSL *glc;
    std::ostringstream src_stream;
    std::string cur_kernel_name;
    Target target;
};


/** Compile one statement into GLSL. */
class CodeGen_GLSL : public CodeGen_C {
public:
    CodeGen_GLSL(std::ostream &s);
    void compile(Stmt stmt,
                 std::string name,
                 const std::vector<GPU_Argument> &args,
                 const Target &target);

    EXPORT static void test();

protected:
    using CodeGen_C::visit;
    std::string print_type(Type type);
    std::string print_name(const std::string &);

    void visit(const FloatImm *);

    void visit(const Cast *);
    void visit(const For *);
    void visit(const Select *);

    void visit(const Max *);
    void visit(const Min *);
    void visit(const Div *);
    void visit(const Mod *);

    void visit(const Load *);
    void visit(const Store *);

    void visit(const Call *);
    void visit(const AssertStmt *);
    void visit(const Broadcast *);

    void visit(const Evaluate *);

private:
    std::string get_vector_suffix(Expr e);

    std::map<std::string, std::string> builtin;
};

}}

#endif
