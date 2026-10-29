// Minimal JSON writer for probe and benchmark reports.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace advp {

class JsonWriter {
 public:
  JsonWriter& begin_object() { return open('{'); }
  JsonWriter& end_object() { return close('}'); }
  JsonWriter& begin_array() { return open('['); }
  JsonWriter& end_array() { return close(']'); }

  JsonWriter& key(const std::string& k) {
    comma();
    string_literal(k);
    out_ += ": ";
    after_key_ = true;
    return *this;
  }
  JsonWriter& value(const std::string& v) {
    comma();
    string_literal(v);
    return *this;
  }
  JsonWriter& value(const char* v) { return value(std::string(v)); }
  JsonWriter& value(bool v) { return raw(v ? "true" : "false"); }
  JsonWriter& value(int v) { return raw(std::to_string(v)); }
  JsonWriter& value(int64_t v) { return raw(std::to_string(v)); }
  JsonWriter& value(uint64_t v) { return raw(std::to_string(v)); }
  JsonWriter& value(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.3f", v);
    return raw(buf);
  }
  template <typename T>
  JsonWriter& field(const std::string& k, T v) {
    return key(k).value(v);
  }

  const std::string& str() const { return out_; }

 private:
  JsonWriter& open(char c) {
    comma();
    out_ += c;
    first_ = true;
    ++depth_;
    return *this;
  }
  JsonWriter& close(char c) {
    --depth_;
    if (!first_) newline();
    out_ += c;
    first_ = false;
    return *this;
  }
  JsonWriter& raw(const std::string& s) {
    comma();
    out_ += s;
    return *this;
  }
  void comma() {
    if (after_key_) {
      after_key_ = false;
      return;
    }
    if (depth_ == 0) return;
    if (!first_) out_ += ',';
    newline();
    first_ = false;
  }
  void newline() {
    out_ += '\n';
    out_.append(static_cast<size_t>(depth_) * 2, ' ');
  }
  void string_literal(const std::string& s) {
    out_ += '"';
    for (char c : s) {
      switch (c) {
        case '"': out_ += "\\\""; break;
        case '\\': out_ += "\\\\"; break;
        case '\n': out_ += "\\n"; break;
        case '\t': out_ += "\\t"; break;
        default:
          if (static_cast<unsigned char>(c) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            out_ += buf;
          } else {
            out_ += c;
          }
      }
    }
    out_ += '"';
  }

  std::string out_;
  int depth_ = 0;
  bool first_ = true;
  bool after_key_ = false;
};

}  // namespace advp
