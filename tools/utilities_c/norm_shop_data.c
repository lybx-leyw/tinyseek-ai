/**
 * 原始格式：
 * {"conversations": [{"role": "user", "content": "[str]"}, {"role": "assistant", "content": "|[str1]|[str2]|[str3]|...[str1]|[str2]|[str3]|"}]}
 * 示例1: "|回形针|一盒|没有|便利贴|一包|没有|红色签字笔|三支|没有|"
 * 示例2: "|生抽|一瓶|海天|猫砂|一包|五公斤 膨润土|"
*/

/** 
 * 目标格式：
 * {"conversations": [{"role": "user", "content": "[str]"}, {"role": "assistant", "content": "你需要[str2][str1]，要求是：[str3]。"}]}
 * 如果是多组，则格式为："第一，你需要[str2][str1]，要求是：[str3]。第二，你需要[str2][str1]，要求是：[str3]。"
 * 如果[str3]是"没有"，则输出："你需要[str2][str1]。"（单组）或"第一，你需要[str2][str1]。"（多组）
 * 如果[str2]是"没有"，则改为"一些"
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DATE_PATH "..\\dataset\\TinySeek_dataset\\lora_shopl.jsonl"
#define WRITE_PATH "..\\dataset\\TinySeek_dataset\\lora_shopl_2.jsonl"

// 中文数字映射
const char* chinese_numbers[] = {"零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"};

// 将数字转换为中文序号（支持1-99）
void number_to_chinese_ordinal(int num, char* output) {
    if (num == 0) {
        strcpy(output, "第零");
        return;
    }
    
    char temp[50] = "第";
    
    if (num <= 10) {
        strcat(temp, chinese_numbers[num]);
    } else if (num < 20) {
        strcat(temp, "十");
        if (num > 10) {
            strcat(temp, chinese_numbers[num - 10]);
        }
    } else if (num < 100) {
        int tens = num / 10;
        int ones = num % 10;
        strcat(temp, chinese_numbers[tens]);
        strcat(temp, "十");
        if (ones > 0) {
            strcat(temp, chinese_numbers[ones]);
        }
    } else {
        sprintf(temp, "第%d", num);
    }
    
    strcpy(output, temp);
}

// 解析一组三个字段
int parse_group(char** current, char* str1, char* str2, char* str3) {
    // 跳过开头的|（如果有）
    if (**current == '|') {
        (*current)++;
    }
    
    // 提取str1
    char* pipe1_end = strchr(*current, '|');
    if (pipe1_end == NULL) return 0;
    int len1 = pipe1_end - *current;
    strncpy(str1, *current, len1);
    str1[len1] = '\0';
    
    // 提取str2
    *current = pipe1_end + 1;
    char* pipe2_end = strchr(*current, '|');
    if (pipe2_end == NULL) return 0;
    int len2 = pipe2_end - *current;
    strncpy(str2, *current, len2);
    str2[len2] = '\0';
    
    // 提取str3
    *current = pipe2_end + 1;
    char* pipe3_end = strchr(*current, '|');
    if (pipe3_end == NULL) return 0;
    int len3 = pipe3_end - *current;
    strncpy(str3, *current, len3);
    str3[len3] = '\0';
    
    *current = pipe3_end + 1;
    return 1;
}

int main()
{
    FILE* fp = fopen(DATE_PATH,"r");
    if (fp == NULL) {
        printf("Cannot open input file: %s\n", DATE_PATH);
        return 1;
    }
    
    FILE* tp = fopen(WRITE_PATH,"w");
    if (tp == NULL) {
        printf("Cannot open output file: %s\n", WRITE_PATH);
        fclose(fp);
        return 1;
    }
    
    char line[4096];
    char target_line[8192];
    int line_count = 0;
    int success_count = 0;
    
    while (fgets(line, sizeof(line), fp) != NULL) {
        line_count++;
        
        // 去除换行符
        line[strcspn(line, "\n")] = 0;
        
        // 提取user内容
        char user_content[500] = "";
        char* user_start = strstr(line, "\"role\": \"user\", \"content\": \"");
        if (user_start) {
            user_start += strlen("\"role\": \"user\", \"content\": \"");
            char* user_end = strstr(user_start, "\"");
            if (user_end) {
                int len = user_end - user_start;
                strncpy(user_content, user_start, len);
                user_content[len] = '\0';
            }
        }
        
        // 提取assistant部分
        char* assis_start = strstr(line, "\"role\": \"assistant\", \"content\": \"");
        if (assis_start == NULL) {
            printf("Line %d conversion failed: no assistant content\n", line_count);
            continue;
        }
        
        assis_start += strlen("\"role\": \"assistant\", \"content\": \"");
        char* assis_end = strstr(assis_start, "\"}");
        if (assis_end == NULL) {
            printf("Line %d conversion failed: invalid assistant format\n", line_count);
            continue;
        }
        
        // 提取完整的assistant content
        int assis_len = assis_end - assis_start;
        char* assis_content = (char*)malloc(assis_len + 1);
        strncpy(assis_content, assis_start, assis_len);
        assis_content[assis_len] = '\0';
        
        // 解析所有组
        char assistant_output[4096] = "";
        char* current = assis_content;
        int item_count = 0;
        int valid = 1;
        
        // 循环解析每一组
        while (*current && valid) {
            char str1[200], str2[200], str3[200];
            
            if (parse_group(&current, str1, str2, str3)) {
                item_count++;
                
                // 处理str2为"没有"的情况
                char quantity[200];
                if (strcmp(str2, "没有") == 0) {
                    strcpy(quantity, "一些");
                } else {
                    strcpy(quantity, str2);
                }
                
                char item_output[500];
                if (strcmp(str3, "没有") == 0) {
                    // 要求是"没有"的情况
                    if (item_count == 1) {
                        // 单组：不需要序号
                        sprintf(item_output, "你需要%s%s。", quantity, str1);
                    } else {
                        // 多组：需要序号
                        char ordinal[50];
                        number_to_chinese_ordinal(item_count, ordinal);
                        sprintf(item_output, "%s，你需要%s%s。", ordinal, quantity, str1);
                    }
                } else {
                    // 有具体要求的情况
                    if (item_count == 1) {
                        // 单组：不需要序号
                        sprintf(item_output, "你需要%s%s，要求是：%s。", quantity, str1, str3);
                    } else {
                        // 多组：需要序号
                        char ordinal[50];
                        number_to_chinese_ordinal(item_count, ordinal);
                        sprintf(item_output, "%s，你需要%s%s，要求是：%s。", ordinal, quantity, str1, str3);
                    }
                }
                
                // 添加到输出
                if (assistant_output[0] == '\0') {
                    strcpy(assistant_output, item_output);
                } else {
                    strcat(assistant_output, item_output);
                }
            } else {
                // 如果当前没有更多完整的组，但还有内容，可能是格式问题
                if (*current) {
                    // 跳过可能存在的额外字符
                    current++;
                }
            }
        }
        
        free(assis_content);
        
        // 检查是否成功（至少有一组）
        if (item_count > 0) {
            sprintf(target_line, "{\"conversations\": ["
                    "{\"role\": \"user\", \"content\": \"请帮我总结购物清单：%s\"}, "
                    "{\"role\": \"assistant\", \"content\": \"好的，总结清单为：%s\"}]}\n", 
                    user_content, assistant_output);
            
            fputs(target_line, tp);
            success_count++;
            printf("Line %d converted successfully (%d items)\n", line_count, item_count);
        } else {
            printf("Line %d conversion failed: no valid groups\n", line_count);
        }
    }
    
    fclose(fp);
    fclose(tp);
    
    printf("\nConversion completed!\n");
    printf("Total lines: %d\n", line_count);
    printf("Successfully converted: %d\n", success_count);
    printf("Output file: %s\n", WRITE_PATH);
    
    return 0;
}