/**
 * 原始格式：
 * {"qid": "[str]", "category": "[str]", "title": "[str]", "desc": "[str1]", "answer": "[str2]"}
 * 目标格式：
 * {"conversations": [{"role": "user", "content": "[str1]"}, {"role": "assistant", "content": "[str2]"}]}
 * 
 * 过滤规则：
 * 1. 如果desc或answer为空，跳过该条记录
 * 2. 如果desc和answer的总中文字符数 > 512，跳过该条记录
 * 3. 如果desc或answer中包含5个及以上连续的英文字母，跳过该条记录
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

#define INPUT_PATH      "..\\dataset\\baidu_dataset\\5hRzSTDh\\baike_qa_train.jsonl"
#define OUTPUT_PATH     "..\\dataset\\TinySeek_dataset\\baike_qa_train.jsonl"
#define MAX_LINE_LEN    40960
#define MAX_CHINESE_CHARS 512  // 中文字符数上限
#define MAX_CONSECUTIVE_ENGLISH 5  // 连续英文字母数量上限
#define FIELD_DESC      "\"desc\": \""
#define FIELD_ANSWER    "\"answer\": \""

/**
 * 检查字符是否为中文字符（UTF-8编码）
 * 中文字符的UTF-8编码范围：0xE4 0xE5 0xE6 0xE7 0xE8 0xE9
 * @param str 指向字符串的指针
 * @return 如果是中文字符返回true，否则返回false
 */
static bool is_chinese_char(const unsigned char *str) {
    // UTF-8中文字符通常占3个字节，首字节范围：0xE4 - 0xE9
    return (str[0] >= 0xE4 && str[0] <= 0xE9 && 
            str[1] >= 0x80 && str[1] <= 0xBF && 
            str[2] >= 0x80 && str[2] <= 0xBF);
}

/**
 * 计算字符串中的中文字符数（UTF-8编码）
 * @param str 输入字符串
 * @return 中文字符数量
 */
static int count_chinese_chars(const char *str) {
    if (str == NULL) return 0;
    
    int count = 0;
    const unsigned char *p = (const unsigned char *)str;
    
    while (*p) {
        if (is_chinese_char(p)) {
            count++;
            p += 3; // 中文字符占3个字节
        } else {
            p++; // 非中文字符占1个字节
        }
    }
    
    return count;
}

/**
 * 检查字符串中是否包含连续指定数量的英文字母
 * @param str 输入字符串
 * @param max_consecutive 最大允许的连续英文字母数量
 * @return 如果包含超过限制的连续英文字母返回true，否则返回false
 */
static bool has_excessive_consecutive_english(const char *str, int max_consecutive) {
    if (str == NULL) return false;
    
    int consecutive_count = 0;
    const unsigned char *p = (const unsigned char *)str;
    
    while (*p) {
        if (is_chinese_char(p)) {
            // 遇到中文字符，重置连续计数
            consecutive_count = 0;
            p += 3;
        } else {
            // 检查是否为英文字母（包括大小写）
            if (isalpha(*p)) {
                consecutive_count++;
                if (consecutive_count >= max_consecutive) {
                    return true; // 发现超过限制的连续英文字母
                }
            } else {
                // 非字母字符（数字、标点、空格等）重置连续计数
                consecutive_count = 0;
            }
            p++;
        }
    }
    
    return false;
}

/**
 * 检查字符是否需要转义（JSON中的特殊字符）
 */
static bool needs_escaping(char c) {
    return c == '\"' || c == '\\' || c == '/' || c == '\b' || 
           c == '\f' || c == '\n' || c == '\r' || c == '\t';
}

/**
 * 转义JSON字符串中的特殊字符
 * @param input 输入字符串
 * @param output 输出缓冲区
 * @param output_size 输出缓冲区大小
 * @return 转义后的字符串长度，-1表示缓冲区不足
 */
static int escape_json_string(const char *input, char *output, size_t output_size) {
    size_t i, j = 0;
    
    for (i = 0; input[i] != '\0' && j < output_size - 1; i++) {
        if (needs_escaping(input[i])) {
            if (j + 1 >= output_size - 1) {
                return -1; // 缓冲区不足
            }
            
            switch (input[i]) {
                case '\"': output[j++] = '\\'; output[j++] = '\"'; break;
                case '\\': output[j++] = '\\'; output[j++] = '\\'; break;
                case '/':  output[j++] = '\\'; output[j++] = '/'; break;
                case '\b': output[j++] = '\\'; output[j++] = 'b'; break;
                case '\f': output[j++] = '\\'; output[j++] = 'f'; break;
                case '\n': output[j++] = '\\'; output[j++] = 'n'; break;
                case '\r': output[j++] = '\\'; output[j++] = 'r'; break;
                case '\t': output[j++] = '\\'; output[j++] = 't'; break;
            }
        } else {
            output[j++] = input[i];
        }
    }
    
    output[j] = '\0';
    return j;
}

/**
 * 提取JSON字符串字段的内容，正确处理转义字符
 * @param start 指向字段值开始位置的指针（字段名和开头的引号之后）
 * @param out_buffer 存储提取内容的缓冲区
 * @param buffer_size 输出缓冲区大小
 * @return 指向结束引号后位置的指针，失败返回NULL
 */
static char* extract_json_string(char* start, char* out_buffer, size_t buffer_size) {
    if (start == NULL || out_buffer == NULL || buffer_size == 0) {
        return NULL;
    }
    
    char *p = start;
    char *out = out_buffer;
    size_t remaining = buffer_size - 1; // 预留一个位置给结束符
    
    while (*p != '\0' && remaining > 0) {
        if (*p == '\\') {
            // 处理转义序列
            p++; // 跳过反斜杠
            if (*p == '\0') break;
            
            switch (*p) {
                case '"':  *out++ = '"'; break;
                case '\\': *out++ = '\\'; break;
                case '/':  *out++ = '/'; break;
                case 'b':  *out++ = '\b'; break;
                case 'f':  *out++ = '\f'; break;
                case 'n':  *out++ = '\n'; break;
                case 'r':  *out++ = '\r'; break;
                case 't':  *out++ = '\t'; break;
                default:   *out++ = *p; break; // 未知转义，保留原字符
            }
            remaining--;
            p++;
        }
        else if (*p == '"') {
            // 找到结束引号
            p++; // 跳过结束引号
            *out = '\0';
            return p;
        }
        else {
            // 普通字符
            *out++ = *p++;
            remaining--;
        }
    }
    
    // 没有找到结束引号或缓冲区不足
    return NULL;
}

int main() {
    FILE *input_file = fopen(INPUT_PATH, "r");
    if (input_file == NULL) {
        printf("Error: Cannot open input file: %s\n", INPUT_PATH);
        return 1;
    }

    FILE *output_file = fopen(OUTPUT_PATH, "w");
    if (output_file == NULL) {
        printf("Error: Cannot open output file: %s\n", OUTPUT_PATH);
        fclose(input_file);
        return 1;
    }

    char line[MAX_LINE_LEN];
    int line_count = 0;
    int success_count = 0;
    int empty_desc_count = 0;
    int empty_answer_count = 0;
    int too_long_count = 0;
    int excessive_english_count = 0;

    while (fgets(line, sizeof(line), input_file)) {
        line_count++;

        // 移除行尾的换行符，但保留行内的换行符（JSON字符串中的\n）
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
        }

        // 查找字段
        char *desc_start = strstr(line, FIELD_DESC);
        char *answer_start = strstr(line, FIELD_ANSWER);

        if (desc_start == NULL || answer_start == NULL) {
            printf("Warning: Line %d missing 'desc' or 'answer' field\n", line_count);
            continue;
        }

        // 提取desc内容
        desc_start += strlen(FIELD_DESC);
        char desc[MAX_LINE_LEN] = {0};
        char *next_pos = extract_json_string(desc_start, desc, sizeof(desc));
        
        if (next_pos == NULL) {
            printf("Warning: Line %d invalid 'desc' format\n", line_count);
            continue;
        }

        // 提取answer内容
        answer_start += strlen(FIELD_ANSWER);
        char answer[MAX_LINE_LEN] = {0};
        next_pos = extract_json_string(answer_start, answer, sizeof(answer));
        
        if (next_pos == NULL) {
            printf("Warning: Line %d invalid 'answer' format\n", line_count);
            continue;
        }

        // 过滤规则1：检查desc或answer是否为空
        if (strlen(desc) == 0) {
            empty_desc_count++;
            continue; // 跳过空desc的记录
        }
        if (strlen(answer) == 0) {
            empty_answer_count++;
            continue; // 跳过空answer的记录
        }

        // 过滤规则2：检查总中文字符数
        int chinese_chars_desc = count_chinese_chars(desc);
        int chinese_chars_answer = count_chinese_chars(answer);
        int total_chinese_chars = chinese_chars_desc + chinese_chars_answer;
        
        if (total_chinese_chars > MAX_CHINESE_CHARS) {
            too_long_count++;
            continue; // 跳过超过长度限制的记录
        }

        // 过滤规则3：检查是否包含过长的连续英文字母
        if (has_excessive_consecutive_english(desc, MAX_CONSECUTIVE_ENGLISH) ||
            has_excessive_consecutive_english(answer, MAX_CONSECUTIVE_ENGLISH)) {
            excessive_english_count++;
            continue; // 跳过包含过长连续英文字母的记录
        }

        // 对输出内容进行JSON转义，确保生成的JSON格式正确
        char escaped_desc[MAX_LINE_LEN * 2] = {0};  // 转义后可能变长，分配更大的缓冲区
        char escaped_answer[MAX_LINE_LEN * 2] = {0};
        
        if (escape_json_string(desc, escaped_desc, sizeof(escaped_desc)) < 0) {
            printf("Warning: Line %d 'desc' too long after escaping\n", line_count);
            continue;
        }
        
        if (escape_json_string(answer, escaped_answer, sizeof(escaped_answer)) < 0) {
            printf("Warning: Line %d 'answer' too long after escaping\n", line_count);
            continue;
        }

        // 写入目标格式
        fprintf(output_file, "{\"conversations\": [");
        fprintf(output_file, "{\"role\": \"user\", \"content\": \"%s\"}, ", escaped_desc);
        fprintf(output_file, "{\"role\": \"assistant\", \"content\": \"%s\"}", escaped_answer);
        fprintf(output_file, "]}\n");
        
        success_count++;
    }

    printf("Conversion completed.\n");
    printf("Total lines processed: %d\n", line_count);
    printf("Successfully converted: %d\n", success_count);
    printf("Filtered out:\n");
    printf("  - Empty desc fields: %d\n", empty_desc_count);
    printf("  - Empty answer fields: %d\n", empty_answer_count);
    printf("  - Too many Chinese chars (>%d): %d\n", MAX_CHINESE_CHARS, too_long_count);
    printf("  - Excessive consecutive English letters (>%d): %d\n", 
           MAX_CONSECUTIVE_ENGLISH, excessive_english_count);
    printf("Total filtered: %d\n", 
           empty_desc_count + empty_answer_count + too_long_count + excessive_english_count);

    fclose(input_file);
    fclose(output_file);
    return 0;
}