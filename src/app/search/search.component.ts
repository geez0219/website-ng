import { Component, OnInit, AfterViewInit}  from '@angular/core';
import { Renderer2, Inject, ElementRef } from '@angular/core';
import { DOCUMENT } from '@angular/common'

@Component({
  selector: 'app-search',
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.css']
})
export class SearchComponent implements OnInit{
  constructor(private renderer: Renderer2, private el: ElementRef, @Inject(DOCUMENT) private _document) {}
  ngOnInit() {
    const s = this.renderer.createElement('script');
    s.type = 'text/javascript';
    s.src = 'https://cse.google.com/cse.js?cx=007435124061301021685:nx5ivx9bz4c';
    s.text="";
    this.renderer.appendChild(this.el.nativeElement, s);
  }
}

// export class SearchComponent implements OnInit, AfterViewInit{
//   constructor(private renderer: Renderer2, private el: ElementRef, @Inject(DOCUMENT) private _document) {}
//   ngOnInit() {
//     const s = this.renderer.createElement('script');
//     s.type = 'text/javascript';
//     s.src = 'https://cse.google.com/cse.js?cx=007435124061301021685:nx5ivx9bz4c';
//     s.text= "";
//     this.renderer.appendChild(this.el.nativeElement, s);
//     // console.log(this.el.nativeElement);
//   }

//   ngAfterViewInit(){
//     console.log(this._document.getElementById("gsc-i-id1"))
//     this._document.getElementById("gsc-i-id1").placeholder = "Search";
//   }
// }