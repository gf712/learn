; hello_world.asm
; Hello world in assembly x86 32-bit
; Author: Gil
; Date  : 26/11/2019


global _start

section .text:
_start:
	mov eax, 0x4		; use the write syscall, checkout unistd_32.h
	mov ebx, 1		; use stdout
	mov ecx, message	; pass message buffer
	mov edx, message_length ; message length
	int 0x80		; invoke syscall
	mov eax, 0x1		; exit the program gracefully
	mov ebx, 0		; return value
	int 0x80		; invoke syscall	


section .data:
	message: db "Hello, world!", 0xA
	message_length equ $-message
