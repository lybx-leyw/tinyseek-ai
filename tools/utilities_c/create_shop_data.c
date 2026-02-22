/** 
 * 目标数据格式：
 * {"conversations": [{"role": "user", "content": "[str]"}, {"role": "assistant", "content": "[str]"}]}
*/

/**
 * 原始格式：
 * [str]@[str]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DATE_PATH "..\\dataset\\TextData_row.txt"
#define WRITE_PATH "..\\dataset\\TinySeek_dataset\\lora_shopl.jsonl"

int main()
{
    FILE* fp = fopen(DATE_PATH,"r");
    FILE* tp = fopen(WRITE_PATH,"w");
    char line[1000];
    char target_line[1000];
    int index=0;
    while (fgets(line,sizeof(line),fp)!=NULL) {
        if (line[strlen(line)-1]=='\n') {
            line[strlen(line)-1] = 0;
        }
        char* pc = line;
        while (*pc!='@') pc++;
        *pc = 0;pc++;
        strcat(target_line,"{\"conversations\": [{\"role\": \"user\", \"content\": \"");
        strcat(target_line,line);
        strcat(target_line,"\"}, {\"role\": \"assistant\", \"content\": \"");
        strcat(target_line,pc);
        strcat(target_line,"\"}]}\n");  
        fputs(target_line,tp);  
        target_line[0] = '\0';
    }
    fclose(fp);
    fclose(tp);
}