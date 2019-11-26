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
    s.src = "https://cse.google.com/cse.js?cx=008491496338527180074:d9p4ksqgel2";
    s.text= "";
    this.renderer.appendChild(this.el.nativeElement, s);
  }
}
