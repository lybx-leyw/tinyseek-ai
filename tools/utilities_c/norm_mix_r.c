/**
 * 原始格式：
 * {"conversations": [{"role": "user", "content": "[str1]"}, {"role": "assistant", "content": "[str2]"}, ...
 *  ,{"role": "user", "content": "[str1]"}, {"role": "assistant", "content": "[str2]"}]}
 * 目标格式不变：
 * {"conversations": [{"role": "user", "content": "[str1]"}, {"role": "assistant", "content": "[str2]"}, ...
 *  ,{"role": "user", "content": "[str1]"}, {"role": "assistant", "content": "[str2]"}]} 
 * 
 * 过滤规则：
 * 1. 如果任一content为空，跳过该条记录
 * 2. 如果所有content的总中文字符数 > 1024，跳过该条记录
 * 3. 如果任一content中包含超过一半的英文字母，跳过整条记录
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

#define INPUT_PATH      "..\\dataset\\minimind_dataset\\r1_mix_1024.jsonl"
#define OUTPUT_PATH     "..\\dataset\\TinySeek_dataset\\r1_mix_1024.jsonl"
#define MAX_LINE_LEN    40960
#define MAX_CHINESE_CHARS 512
#define MAX_CONTENT_LEN 4096
#define MAX_CONVERSATIONS 100

// 判断字符串是否为空或仅含空白
bool is_blank(const char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    return *s == '\0';
}

// 统计中文字符数（UTF-8编码）
int count_chinese_chars(const char *s) {
    int count = 0;
    while (*s) {
        unsigned char c = (unsigned char)*s;
        // 中文字符通常为三字节UTF-8，首字节范围[0xE4,0xE9][0xE9,0xEF]等
        if ((c & 0xF0) == 0xE0) { // 三字节UTF-8的首字节特征
            count++;
            s += 3;
        } else {
            s++;
        }
    }
    return count;
}

// 统计总字符数和英文字母数
void count_chars_and_letters(const char *s, int *total_chars, int *letters) {
    *total_chars = 0;
    *letters = 0;
    while (*s) {
        unsigned char c = (unsigned char)*s;
        (*total_chars)++;
        if (c <= 0x7F) { // ASCII
            if (isalpha(c)) (*letters)++;
            s++;
        } else if (c <= 0xDF) { // 2字节
            s += 2;
        } else if (c <= 0xEF) { // 3字节
            s += 3;
        } else { // 4字节
            s += 4;
        }
    }
}

// 找到字符串结束位置，处理转义字符
const char* find_string_end(const char *start) {
    const char *p = start;
    while (*p) {
        if (*p == '\\') {
            // 转义字符，跳过下一个字符
            if (*(p+1)) p += 2;
            else p++;
        } else if (*p == '"') {
            // 找到未转义的结束引号
            return p;
        } else {
            p++;
        }
    }
    return NULL; // 没找到结束引号
}

// 从JSON行中提取所有content的值（正确处理转义）
int extract_contents(const char *line, char contents[][MAX_CONTENT_LEN], int max_cnt) {
    int cnt = 0;
    const char *p = line;
    
    while (*p && cnt < max_cnt) {
        // 查找 "content"
        p = strstr(p, "\"content\"");
        if (!p) break;
        
        p += 9; // 跳过 "content"
        
        // 跳过空白字符
        while (*p && isspace((unsigned char)*p)) p++;
        
        // 检查冒号
        if (*p != ':') {
            p++;
            continue;
        }
        p++;
        
        // 跳过空白字符
        while (*p && isspace((unsigned char)*p)) p++;
        
        // 检查字符串开始
        if (*p != '"') {
            p++;
            continue;
        }
        p++; // 跳过起始引号
        
        // 找到字符串结束位置（处理转义）
        const char *end = find_string_end(p);
        if (!end) break;
        
        // 提取内容（不包含引号）
        size_t len = end - p;
        if (len >= MAX_CONTENT_LEN) len = MAX_CONTENT_LEN - 1;
        strncpy(contents[cnt], p, len);
        contents[cnt][len] = '\0';
        
        // 处理转义字符：将 \" 替换为 "，\\ 替换为 \ 等
        char *src = contents[cnt];
        char *dst = contents[cnt];
        while (*src) {
            if (*src == '\\') {
                src++;
                if (*src == '"' || *src == '\\' || *src == '/' || 
                    *src == 'b' || *src == 'f' || *src == 'n' || 
                    *src == 'r' || *src == 't') {
                    // 保留转义后的字符
                    *dst++ = *src++;
                } else {
                    // 未知转义，保留原样
                    *dst++ = '\\';
                }
            } else {
                *dst++ = *src++;
            }
        }
        *dst = '\0';
        
        cnt++;
        p = end + 1; // 继续往后找
    }
    
    return cnt;
}

int main() {
    FILE *fin = fopen(INPUT_PATH, "r");
    if (!fin) {
        perror("Failed to open input file");
        return 1;
    }
    FILE *fout = fopen(OUTPUT_PATH, "w");
    if (!fout) {
        perror("Failed to open output file");
        fclose(fin);
        return 1;
    }

    char line[MAX_LINE_LEN];
    int line_num = 0;
    int kept = 0, skipped = 0;

    while (fgets(line, sizeof(line), fin)) {
        line_num++;
        // 去除换行符
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';

        // 提取content
        char contents[MAX_CONVERSATIONS][MAX_CONTENT_LEN];
        int content_cnt = extract_contents(line, contents, MAX_CONVERSATIONS);
        
        if (content_cnt == 0) {
            printf("警告：第%d行未找到任何content字段\n", line_num);
            skipped++;
            continue;
        }

        // 规则1：检查空content
        bool has_empty = false;
        for (int i = 0; i < content_cnt; i++) {
            if (is_blank(contents[i])) {
                has_empty = true;
                break;
            }
        }
        if (has_empty) {
            skipped++;
            continue;
        }

        // 规则2：中文字符数限制
        int total_chinese = 0;
        for (int i = 0; i < content_cnt; i++) {
            total_chinese += count_chinese_chars(contents[i]);
            if (total_chinese > MAX_CHINESE_CHARS) break;
        }
        if (total_chinese > MAX_CHINESE_CHARS) {
            skipped++;
            continue;
        }

        // 规则3：英文字母比例检查
        bool too_many_letters = false;
        for (int i = 0; i < content_cnt; i++) {
            int total_chars, letters;
            count_chars_and_letters(contents[i], &total_chars, &letters);
            if (total_chars > 0 && letters * 2 > total_chars) {
                too_many_letters = true;
                break;
            }
        }
        if (too_many_letters) {
            skipped++;
            continue;
        }

        // 保留
        fprintf(fout, "%s\n", line);
        kept++;
    }

    printf("处理完成：总行数 %d，保留 %d，跳过 %d\n", line_num, kept, skipped);
    fclose(fin);
    fclose(fout);
    return 0;
}